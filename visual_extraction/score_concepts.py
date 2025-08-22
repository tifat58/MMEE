#!/usr/bin/env python3
import os, json, argparse
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

def load_artifacts(features_dir: str):
    C = torch.load(os.path.join(features_dir, "C.pt"))              # J x D
    tokens = json.load(open(os.path.join(features_dir, "tokens.json")))
    Fg = torch.load(os.path.join(features_dir, "Fg.pt"))            # N x D
    paths = json.load(open(os.path.join(features_dir, "paths.json")))
    # Optional files
    labels = None
    labels_path = os.path.join(features_dir, "labels.pt")
    if os.path.isfile(labels_path):
        labels = torch.load(labels_path)                            # N
    Fmap = None
    fmp = os.path.join(features_dir, "Fmap.pt")
    if os.path.isfile(fmp):
        Fmap = torch.load(fmp)                                      # N x Hf x Wf x D
    return Fg, Fmap, C, tokens, paths, labels

def l2norm(x: torch.Tensor, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def global_topk(Fg: torch.Tensor, C: torch.Tensor, topk: int = 5, chunk: int = 2048):
    """
    Fg: N x D (assumed roughly normalized; we re-normalize just in case)
    C:  J x D
    Returns:
      top_scores: N x topk
      top_idx:    N x topk  (indices into concepts)
    """
    Fg = l2norm(Fg.float())
    C = l2norm(C.float())
    N, D = Fg.shape
    J = C.shape[0]
    # chunk over concepts to save memory
    scores_acc = []
    for start in range(0, J, chunk):
        end = min(J, start+chunk)
        # (N x D) @ (D x jchunk) -> N x jchunk
        scores_acc.append(Fg @ C[start:end].T)
    scores = torch.cat(scores_acc, dim=1)   # N x J
    top_scores, top_idx = torch.topk(scores, k=min(topk, J), dim=1)
    return top_scores.cpu(), top_idx.cpu()

@torch.no_grad()
def region_topk(Fmap: torch.Tensor, C: torch.Tensor, topk: int = 5,
                pool: str = "max", alpha: float = 10.0, chunk: int = 1024):
    """
    Fmap: N x Hf x Wf x D
    C:    J x D
    pool: 'max' or 'noisy_or'
    Returns:
      top_scores: N x topk
      top_idx:    N x topk
      top_coords: N x topk x 2   (y,x) of the best patch per top concept (for 'max'),
                  or the argmax-patch under pooled scores (for 'noisy_or', we still report max-patch)
    """
    Fmap = l2norm(Fmap.float(), dim=-1)
    C = l2norm(C.float())
    N, Hf, Wf, D = Fmap.shape
    J = C.shape[0]
    R = Hf * Wf

    # flatten spatial to (N, R, D)
    Fflat = Fmap.view(N, R, D)
    # compute sim per patch in chunks over concepts
    # we'll accumulate either max over patches or noisy-or aggregation
    if pool == "max":
        pooled = torch.empty(N, J, device=Fmap.device, dtype=torch.float32)
        argmax_r = torch.empty(N, J, device=Fmap.device, dtype=torch.long)
    else:
        # noisy-or via sigmoid(alpha*s); accumulate product over patches
        prod_acc = torch.ones(N, J, device=Fmap.device, dtype=torch.float32)
        # for reporting coords, we still track max-patch
        pooled_max = torch.empty(N, J, device=Fmap.device, dtype=torch.float32)
        argmax_r   = torch.empty(N, J, device=Fmap.device, dtype=torch.long)

    for start in tqdm(range(0, J, chunk), desc="Region scoring"):
        end = min(J, start+chunk)
        Cslice = C[start:end]                 # jchunk x D
        # (N,R,D) @ (D,jchunk) -> (N,R,jchunk)
        S = torch.matmul(Fflat, Cslice.T)

        if pool == "max":
            Smax, ridx = torch.max(S, dim=1)  # N x jchunk, N x jchunk
            pooled[:, start:end] = Smax
            argmax_r[:, start:end] = ridx
        else:
            # noisy-or
            P = torch.sigmoid(alpha * S)                  # N x R x jchunk
            one_minus = 1.0 - P.clamp(0.0, 1.0)
            prod = torch.exp(torch.sum(torch.log(one_minus + 1e-6), dim=1))  # N x jchunk
            z = 1.0 - prod.clamp(0.0, 1.0)
            if start == 0:
                pooled_max[:, start:end], arg = torch.max(S, dim=1)
            else:
                # track max for coords across chunks too
                curmax, arg = torch.max(S, dim=1)
                pooled_max[:, start:end] = curmax
            # store prod in a temp buffer then assign to pooled at the end
            if start == 0:
                pooled_noisyor = z
            else:
                pooled_noisyor = torch.cat([pooled_noisyor, z], dim=1)
            argmax_r[:, start:end] = arg

    if pool == "max":
        pooled_scores = pooled
    else:
        pooled_scores = pooled_noisyor

    # topk concepts per image
    top_scores, top_idx = torch.topk(pooled_scores, k=min(topk, J), dim=1)
    # coords for the best patch of each selected concept
    coords = []
    for n in range(N):
        this_coords = []
        for jrank in range(top_idx.shape[1]):
            j = top_idx[n, jrank].item()
            r = argmax_r[n, j].item()
            y, x = divmod(r, Wf)
            this_coords.append([int(y), int(x)])
        coords.append(this_coords)
    top_coords = torch.tensor(coords, dtype=torch.long)
    return top_scores.cpu(), top_idx.cpu(), top_coords

def write_outputs(paths: List[str],
                  tokens: List[str],
                  labels: torch.Tensor,
                  top_scores: torch.Tensor,
                  top_idx: torch.Tensor,
                  top_coords: torch.Tensor = None,
                  out_json: str = "topk.json",
                  out_tsv: str = "topk.tsv"):
    """
    Save a human-friendly JSON and TSV.
    """
    N, K = top_idx.shape
    records = []
    for i in range(N):
        entry = {
            "path": paths[i],
            "label": int(labels[i]) if labels is not None else None,
            "topk": []
        }
        for r in range(K):
            cid = int(top_idx[i, r])
            item = {
                "concept": tokens[cid],
                "score": float(top_scores[i, r])
            }
            if top_coords is not None:
                item["patch_yx"] = [int(top_coords[i, r, 0]), int(top_coords[i, r, 1])]
            entry["topk"].append(item)
        records.append(entry)
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)

    # TSV: path \t label \t concept_1:score \t concept_2:score ...
    with open(out_tsv, "w") as f:
        header = ["path", "label"] + [f"c{r+1}:score" for r in range(K)]
        f.write("\t".join(header) + "\n")
        for rec in records:
            row = [rec["path"], str(rec["label"])]
            row += [f'{x["concept"]}:{x["score"]:.4f}' for x in rec["topk"]]
            f.write("\t".join(row) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="Directory containing Fg.pt, Fmap.pt, C.pt, tokens.json, paths.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=["global","regions"], default="global")
    ap.add_argument("--pool", choices=["max","noisy_or"], default="max", help="Pooling for regions mode")
    ap.add_argument("--alpha", type=float, default=10.0, help="Sharpness for noisy_or (sigmoid(alpha*s))")
    ap.add_argument("--chunk", type=int, default=1024, help="Concept chunk for matmul")
    ap.add_argument("--suffix", default="", help="Suffix string added to output filenames")
    args = ap.parse_args()

    Fg, Fmap, C, tokens, paths, labels = load_artifacts(args.features_dir)

    if args.mode == "global":
        top_scores, top_idx = global_topk(Fg, C, topk=args.topk, chunk=args.chunk)
        out_json = os.path.join(args.features_dir, f"topk_global{args.suffix}.json")
        out_tsv  = os.path.join(args.features_dir, f"topk_global{args.suffix}.tsv")
        write_outputs(paths, tokens, labels, top_scores, top_idx, None, out_json, out_tsv)
        print(f"Saved: {out_json}\n       {out_tsv}")

    else:
        if Fmap is None:
            raise FileNotFoundError("Fmap.pt not found in features-dir. Re-run extractor that saves patch grid features.")
        top_scores, top_idx, top_coords = region_topk(Fmap, C, topk=args.topk,
                                                      pool=args.pool, alpha=args.alpha, chunk=args.chunk)
        out_json = os.path.join(args.features_dir, f"topk_regions_{args.pool}{args.suffix}.json")
        out_tsv  = os.path.join(args.features_dir, f"topk_regions_{args.pool}{args.suffix}.tsv")
        write_outputs(paths, tokens, labels, top_scores, top_idx, top_coords, out_json, out_tsv)
        print(f"Saved: {out_json}\n       {out_tsv}")

if __name__ == "__main__":
    main()
