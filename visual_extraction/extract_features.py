#!/usr/bin/env python3
import os, json, argparse
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import clip  # pip install git+https://github.com/openai/CLIP.git

# ----------------------------
#   CLIP feature extraction
# ----------------------------
@torch.no_grad()
def extract_patch_grid(clip_model, images: torch.Tensor):
    """
    Returns:
      Fmap: B x Hf x Wf x Dtok  (L2-normalized patch/token features)
      Fg:   B x Dtok            (L2-normalized global CLS feature)
    Handles dtype casting and works for ViT or RN CLIP.
    """
    v = clip_model.visual
    device = images.device

    # Match image dtype to visual weights (fp16 on GPU by default)
    target_dtype = v.conv1.weight.dtype if hasattr(v, "conv1") else images.dtype
    images = images.to(device=device, dtype=target_dtype)

    # ---- ResNet fallback (no transformer tokens) ----
    if not hasattr(v, "transformer"):
        Fg = clip_model.encode_image(images)              # B x Dtok
        Fg = Fg / Fg.norm(dim=-1, keepdim=True)
        B, Dtok = Fg.shape
        Fmap = Fg.view(B, 1, 1, Dtok).expand(B, 14, 14, Dtok).contiguous()
        return Fmap, Fg

    # ---- ViT path ----
    # 1) Patch embedding conv
    x = v.conv1(images)                                   # B, Dconv, Hf, Wf
    B, Dconv, Hf, Wf = x.shape

    # 2) Flatten patches to sequence
    x = x.reshape(B, Dconv, Hf * Wf).permute(0, 2, 1)     # B, L, Dconv
    L = x.shape[1]

    # 3) Prepend CLS token
    cls = v.class_embedding.to(x.dtype)
    cls_tok = cls + torch.zeros(B, 1, Dconv, dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tok, x], dim=1)                    # B, 1+L, Dconv

    # 4) Positional embeddings
    pos = v.positional_embedding.to(x.dtype)
    if pos.shape[0] != x.shape[1]:
        raise RuntimeError(
            f"Positional embedding mismatch: pos={pos.shape[0]} vs seq={x.shape[1]}. "
            "Use the preprocess from clip.load() (e.g., 224x224 for ViT-B/16)."
        )
    x = x + pos

    # 5) Transformer
    x = v.ln_pre(x)
    x = x.permute(1, 0, 2)                                # (1+L), B, Dconv
    x = v.transformer(x)
    x = x.permute(1, 0, 2)                                # B, (1+L), Dconv
    x = v.ln_post(x)
    if v.proj is not None:
        x = x @ v.proj                                    # project Dconv -> Dtok

    # 6) Split CLS vs patches
    cls_feat   = x[:, 0]                                  # B, Dtok
    patch_feat = x[:, 1:]                                 # B, L, Dtok
    Dtok = patch_feat.shape[-1]                           # <-- NEW: current token dim

    # 7) Reshape with the actual Hf, Wf and **current** Dtok
    patch_feat = patch_feat.view(B, Hf, Wf, Dtok)

    # 8) L2-normalize
    Fg   = cls_feat / cls_feat.norm(dim=-1, keepdim=True)     # B, Dtok
    Fmap = patch_feat / patch_feat.norm(dim=-1, keepdim=True) # B, Hf, Wf, Dtok
    return Fmap, Fg


@torch.no_grad()
def encode_texts(clip_model, tokens: List[str], batch_size: int = 256, device="cuda"):
    outs = []
    for i in tqdm(range(0, len(tokens), batch_size), desc="Text encode"):
        chunk = tokens[i:i + batch_size]
        tok = clip.tokenize(chunk).to(device)
        feats = clip_model.encode_text(tok)               # B x D
        feats = feats / feats.norm(dim=-1, keepdim=True)
        outs.append(feats.float().cpu())
    return torch.cat(outs, dim=0)                         # J x D


# ----------------------------
#             Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, help="Path to cub_images/ (folders per class)")
    ap.add_argument("--vocab-json",  required=True, help="V2C vocabulary JSON (list[str] OR dict[class]->list[str])")
    ap.add_argument("--output",      required=True, help="Output directory for .pt/.json files")
    ap.add_argument("--model",       default="ViT-B/16", choices=["ViT-B/16","ViT-L/14","RN50","RN101"])
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-fp16",   action="store_true", help="Save features in float16 to reduce disk")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load CLIP & preprocess
    device = args.device
    clip_model, preprocess = clip.load(args.model, device=device)
    clip_model.eval()

    # Dataset via folder-per-class layout
    ds = datasets.ImageFolder(root=args.images_root, transform=preprocess)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Persist class mappings and paths/labels
    class_to_idx = ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(args.output, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(os.path.join(args.output, "idx_to_class.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)

    all_paths = [s[0] for s in ds.samples]
    labels    = torch.tensor([s[1] for s in ds.samples], dtype=torch.long)
    N = len(ds)

    # Peek one batch to know shapes
    images0, _ = next(iter(dl))
    images0 = images0.to(device)
    Fmap0, Fg0 = extract_patch_grid(clip_model, images0)
    _, Hf, Wf, D = Fmap0.shape

    # Allocate feature tensors
    dtype_store = torch.float16 if args.save_fp16 else torch.float32
    Fg_all   = torch.empty((N, D),             dtype=dtype_store)
    Fmap_all = torch.empty((N, Hf, Wf, D),     dtype=dtype_store)

    # Encode all images
    n_seen = 0
    for images, _ in tqdm(dl, desc="Image encode"):
        B = images.size(0)
        images = images.to(device, non_blocking=True)
        Fmap, Fg = extract_patch_grid(clip_model, images)
        Fg_all[n_seen:n_seen+B]   = Fg.to(dtype_store).cpu()
        Fmap_all[n_seen:n_seen+B] = Fmap.to(dtype_store).cpu()
        n_seen += B

    # Save image features + labels + paths
    torch.save(Fg_all,   os.path.join(args.output, "Fg.pt"))
    torch.save(Fmap_all, os.path.join(args.output, "Fmap.pt"))
    torch.save(labels,   os.path.join(args.output, "labels.pt"))
    with open(os.path.join(args.output, "paths.json"), "w") as f:
        json.dump(all_paths, f, indent=2)

    # Load vocabulary (list OR dict[class]->list) and flatten to unique tokens
    with open(args.vocab_json, "r") as f:
        raw = json.load(f)

    seen, tokens = set(), []
    if isinstance(raw, dict):
        for _, lst in raw.items():          # class2concepts.json → flatten
            for t in lst:
                t = str(t).strip()
                if t and t not in seen:
                    seen.add(t)
                    tokens.append(t)
    elif isinstance(raw, list):
        for t in raw:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                tokens.append(t)
    else:
        raise ValueError("vocab-json must be a list[str] or dict[str, list[str]]")

    if len(tokens) == 0:
        raise ValueError("No valid tokens found after cleaning your vocab JSON.")

    # Encode vocab with the SAME CLIP text tower
    C = encode_texts(clip_model, tokens, batch_size=256, device=device)  # J x D, L2-normalized
    torch.save(C, os.path.join(args.output, "C.pt"))
    with open(os.path.join(args.output, "tokens.json"), "w") as f:
        json.dump(tokens, f, indent=2)

    print("Done.")
    print(f" Images:       {N}")
    print(f" Global  Fg:   {tuple(Fg_all.shape)} → {os.path.join(args.output, 'Fg.pt')}")
    print(f" Patch   Fmap: {tuple(Fmap_all.shape)} → {os.path.join(args.output, 'Fmap.pt')}")
    print(f" Labels:       {tuple(labels.shape)} → {os.path.join(args.output, 'labels.pt')}")
    print(f" Concepts  C:  {tuple(C.shape)} → {os.path.join(args.output, 'C.pt')}")
    print(f" Tokens:       {len(tokens)} → {os.path.join(args.output, 'tokens.json')}")
    print(f" Class map:    {os.path.join(args.output, 'class_to_idx.json')}")
    print(f" Paths:        {os.path.join(args.output, 'paths.json')}")

if __name__ == "__main__":
    main()
