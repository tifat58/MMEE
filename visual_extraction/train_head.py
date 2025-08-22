#!/usr/bin/env python3
import os, json, argparse, random
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def load_artifacts(features_dir: str):
    C = torch.load(os.path.join(features_dir, "C.pt"))              # J x Dtok
    tokens = json.load(open(os.path.join(features_dir, "tokens.json")))
    Fmap = torch.load(os.path.join(features_dir, "Fmap.pt"))        # N x Hf x Wf x Dtok (may be fp16)
    labels = torch.load(os.path.join(features_dir, "labels.pt"))    # N
    paths = json.load(open(os.path.join(features_dir, "paths.json")))
    idx_to_class = json.load(open(os.path.join(features_dir, "idx_to_class.json")))
    return Fmap, labels, C, tokens, paths, idx_to_class

# ----------------------------
# Streaming concept aggregation (compute z)
# ----------------------------
@torch.no_grad()
def compute_z_from_patchgrid(
    Fmap: torch.Tensor,        # N x Hf x Wf x Dtok  (CPU tensor; can be fp16)
    C: torch.Tensor,           # J x Dtok            (float32 recommended)
    pool: str = "noisy_or",    # 'max' | 'noisy_or'
    alpha: float = 10.0,
    concept_chunk: int = 256,  # #concepts per chunk
    region_chunk: int = 64,    # #patches per chunk (Hf*Wf can be chunked)
    img_batch: int = 64,       # #images per mini-batch
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = None
) -> torch.Tensor:
    """
    Streaming computation of image-level concept vector Z (N x J) from patch grid features.
    Chunks over images, concepts, and regions to avoid OOM.
    """
    assert pool in ("max", "noisy_or")
    N, Hf, Wf, D = Fmap.shape
    J, D2 = C.shape
    assert D == D2, f"D mismatch: Fmap D={D} vs C D={D2}"

    # Normalize concepts once on device
    C = l2norm(C.float()).to(device)

    # Output on CPU; we fill it in batches
    Z_cpu = torch.empty((N, J), dtype=torch.float32)

    # Iterate over images in mini-batches (stay on CPU, move slices to GPU)
    for i0 in tqdm(range(0, N, img_batch), desc=f"Compute z [{pool}] (imgs)"):
        i1 = min(N, i0 + img_batch)
        # (nb, Hf, Wf, D) -> normalize per-patch on device
        Fmb = Fmap[i0:i1]  # still on CPU, possibly fp16
        nb = Fmb.shape[0]
        R = Hf * Wf

        # Flatten spatial: (nb, R, D)
        Fmb = Fmb.view(nb, R, D).to(device)
        Fmb = l2norm(Fmb.float(), dim=-1)

        # Temp storage for this image batch on device
        Zb = torch.empty((nb, J), dtype=torch.float32, device=device)

        # Iterate over concept chunks
        for j0 in tqdm(range(0, J, concept_chunk), leave=False, desc="concept chunks"):
            j1 = min(J, j0 + concept_chunk)
            Cslice = C[j0:j1]  # (jc, D)
            jc = Cslice.shape[0]

            if pool == "max":
                # start very small values for max
                Smax = torch.full((nb, jc), -1e9, dtype=torch.float32, device=device)

                # iterate over region chunks
                for r0 in range(0, R, region_chunk):
                    r1 = min(R, r0 + region_chunk)
                    # (nb, rc, D) @ (D, jc) -> (nb, rc, jc)
                    S = torch.matmul(Fmb[:, r0:r1, :], Cslice.T)  # float32
                    # max over regions in this chunk
                    S_chunk_max, _ = torch.max(S, dim=1)          # (nb, jc)
                    Smax = torch.maximum(Smax, S_chunk_max)

                Zb[:, j0:j1] = Smax  # scores in [-1,1] roughly

            else:
                # noisy_or: accumulate sum of log(1 - sigmoid(alpha*S)) over regions
                # initialize running log product sum: log Π (1-p) = Σ log(1-p)
                log_prod_sum = torch.zeros((nb, jc), dtype=torch.float32, device=device)

                for r0 in range(0, R, region_chunk):
                    r1 = min(R, r0 + region_chunk)
                    S = torch.matmul(Fmb[:, r0:r1, :], Cslice.T)          # (nb, rc, jc)
                    # sigmoid in float32, clamp for stability
                    P = torch.sigmoid(alpha * S).to(torch.float32).clamp_(0.0, 1.0)
                    one_minus = (1.0 - P).clamp_(1e-6, 1.0)               # avoid log(0)
                    log_prod_sum += torch.sum(torch.log(one_minus), dim=1)  # (nb, jc)

                z = 1.0 - torch.exp(log_prod_sum)                         # (nb, jc)
                Zb[:, j0:j1] = z.clamp_(0.0, 1.0)

            # free some memory
            del Cslice

        # bring batch back to CPU
        Z_cpu[i0:i1] = Zb.cpu()
        del Fmb, Zb
        torch.cuda.empty_cache()

    if save_path:
        torch.save(Z_cpu, save_path)
    return Z_cpu

# ----------------------------
# Splitting utilities
# ----------------------------
def make_fewshot_indices(labels: torch.Tensor, shots: int, val_per_class: int, seed: int = 0):
    set_seed(seed)
    labels_list = labels.cpu().tolist()
    n_classes = max(labels_list) + 1
    per_class = [[] for _ in range(n_classes)]
    for i, y in enumerate(labels_list):
        per_class[y].append(i)
    for y in range(n_classes):
        random.shuffle(per_class[y])

    idx_train, idx_val, idx_test = [], [], []
    for y in range(n_classes):
        items = per_class[y]
        tr = items[:shots]
        va = items[shots:shots + val_per_class]
        te = items[shots + val_per_class:]
        if len(va) == 0 and len(te) > 0:
            va, te = te[:1], te[1:]
        idx_train += tr
        idx_val   += va
        idx_test  += te
    random.shuffle(idx_train); random.shuffle(idx_val); random.shuffle(idx_test)
    return idx_train, idx_val, idx_test

def make_fullsplit_indices(labels: torch.Tensor, val_ratio: float = 0.1, seed: int = 0):
    set_seed(seed)
    labels_list = labels.cpu().tolist()
    n_classes = max(labels_list) + 1
    per_class = [[] for _ in range(n_classes)]
    for i, y in enumerate(labels_list):
        per_class[y].append(i)
    idx_train, idx_val = [], []
    for y in range(n_classes):
        items = per_class[y]
        random.shuffle(items)
        k = max(1, int(len(items) * val_ratio))
        idx_val += items[:k]
        idx_train += items[k:]
    idx_test = list(idx_val)
    random.shuffle(idx_train); random.shuffle(idx_val); random.shuffle(idx_test)
    return idx_train, idx_val, idx_test

# ----------------------------
# Linear head training
# ----------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)

def train_linear_head(
    Z: torch.Tensor, labels: torch.Tensor,
    idx_train: List[int], idx_val: List[int], idx_test: List[int],
    epochs: int = 100, lr: float = 1e-3, weight_decay: float = 0.0,
    batch_size: int = 256, device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = None
):
    Z = Z.to(device)
    y = labels.to(device).long()
    n_classes = int(labels.max().item() + 1)

    model = LinearHead(Z.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    def make_loader(idxs, shuffle=False):
        Zd = Z[idxs]; yd = y[idxs]
        ds = TensorDataset(Zd, yd)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    tr_loader = make_loader(idx_train, shuffle=True)
    va_loader = make_loader(idx_val,   shuffle=False)
    te_loader = make_loader(idx_test,  shuffle=False)

    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for z_b, y_b in tr_loader:
            logits = model(z_b)
            loss = criterion(logits, y_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # eval
        model.eval()
        def eval_loader(loader):
            acc, n = 0.0, 0
            with torch.no_grad():
                for z_b, y_b in loader:
                    acc += (model(z_b).argmax(1) == y_b).float().sum().item()
                    n += y_b.numel()
            return acc / max(1, n)

        tr_acc = eval_loader(tr_loader)
        va_acc = eval_loader(va_loader)
        te_acc = eval_loader(te_loader)
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{ep:03d}] train_acc={tr_acc*100:.2f}  val_acc={va_acc*100:.2f}  test_acc={te_acc*100:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        torch.save({"state_dict": model.state_dict(),
                    "in_dim": Z.shape[1],
                    "n_classes": int(labels.max().item() + 1)}, save_path)

    # final eval
    tr = eval_loader(tr_loader); va = eval_loader(va_loader); te = eval_loader(te_loader)
    return tr, va, te

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="dir with Fmap.pt, C.pt, labels.pt, tokens.json, idx_to_class.json")
    ap.add_argument("--pool", choices=["max","noisy_or"], default="noisy_or")
    ap.add_argument("--alpha", type=float, default=10.0, help="sharpness for noisy_or")
    ap.add_argument("--chunk", type=int, default=256, help="concept chunk size")
    ap.add_argument("--region-chunk", type=int, default=64, help="region (patch) chunk size")
    ap.add_argument("--img-batch", type=int, default=64, help="image batch for Z computation")
    ap.add_argument("--save-z", action="store_true", help="save computed Z.pt")
    ap.add_argument("--seed", type=int, default=0)

    # Split options
    sp = ap.add_subparsers(dest="split", required=True)
    fs = sp.add_parser("fewshot")
    fs.add_argument("--shots", type=int, default=4, help="train shots per class")
    fs.add_argument("--val-per-class", type=int, default=8)

    full = sp.add_parser("full")
    full.add_argument("--val-ratio", type=float, default=0.1)

    # Train options (global)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-head", default=None, help="path to save trained linear head")

    args = ap.parse_args()
    set_seed(args.seed)

    # Load data
    Fmap, labels, C, tokens, paths, idx_to_class = load_artifacts(args.features_dir)
    device = args.device

    # Compute Z (streaming, OOM-safe)
    z_path = os.path.join(args.features_dir, f"Z_{args.pool}.pt")
    if os.path.isfile(z_path) and args.save_z:
        print(f"Found existing Z at {z_path}, loading...")
        Z = torch.load(z_path)
    else:
        Z = compute_z_from_patchgrid(
            Fmap, C,
            pool=args.pool, alpha=args.alpha,
            concept_chunk=args.chunk,
            region_chunk=args.region_chunk,
            img_batch=args.img_batch,
            device=device,
            save_path=z_path if args.save_z else None
        )

    # Build splits
    if args.split == "fewshot":
        idx_tr, idx_va, idx_te = make_fewshot_indices(labels, shots=args.shots, val_per_class=args.val_per_class, seed=args.seed)
        print(f"Few-shot splits: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")
    else:
        idx_tr, idx_va, idx_te = make_fullsplit_indices(labels, val_ratio=args.val_ratio, seed=args.seed)
        print(f"Full-data splits: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # Train linear head
    tr, va, te = train_linear_head(
        Z, labels, idx_tr, idx_va, idx_te,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, device=device, save_path=args.save_head
    )

    print(f"Final accuracy  |  train: {tr*100:.2f}  val: {va*100:.2f}  test: {te*100:.2f}")
    if args.save_head:
        print(f"Saved linear head to: {args.save_head}")

if __name__ == "__main__":
    main()
