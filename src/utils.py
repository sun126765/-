import os, math, json
from typing import Dict, List, Optional
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_jsonl(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def plot_curve(points: List[Dict], x_key: str, y_key: str, out_png: str, title: Optional[str]=None):
    xs = [p[x_key] for p in points]
    ys = [p[y_key] for p in points]
    plt.figure()
    plt.plot(xs, ys)
    if title:
        plt.title(title)
    plt.xlabel(x_key); plt.ylabel(y_key)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def subsequent_mask(size: int, device=None):
    # Causal mask: [1, size, size]
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    return mask  # True means masked

def make_pad_mask(lengths, max_len=None, device=None):
    # lengths: [B], return [B, 1, 1, L] broadcasting mask (True means masked)
    if max_len is None:
        max_len = int(lengths.max())
    range_row = torch.arange(max_len, device=device).unsqueeze(0)  # [1, L]
    mask = range_row >= lengths.unsqueeze(1)  # [B, L]
    return mask.unsqueeze(1).unsqueeze(1)     # [B, 1, 1, L]
