import os, math, json, argparse
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from .models import EncoderClassifier
from .utils import set_seed, save_jsonl, plot_curve, make_pad_mask

@dataclass
class Config:
    model_dim: int = 128
    ffn_dim: int = 512
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 5
    seed: int = 42
    max_len: int = 256
    save_dir: str = "results/agnews_encoder"
    model_name: str = "bert-base-uncased"

def collate_fn(batch, tokenizer, max_len):
    texts = [x['text'] for x in batch]
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    out = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = out['input_ids']
    lengths = (out['attention_mask']).sum(dim=1)
    return input_ids, lengths, labels

def masked_mean(x, lengths):
    # x: [B, L, D]
    B, L, D = x.shape
    mask = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]
    mask = mask.float().unsqueeze(-1)  # [B, L, 1]
    summed = (x * mask).sum(dim=1)    # [B, D]
    denom = lengths.clamp_min(1).unsqueeze(1).float()
    return summed / denom

def train_one_epoch(model, loader, optim, device):
    model.train()
    total, correct, n = 0.0, 0, 0
    crit = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids, lengths, labels = [x.to(device) for x in batch]
        src_mask = make_pad_mask(lengths, max_len=input_ids.size(1), device=device)  # [B,1,1,L]

        # DataParallel 会在这里自动切 batch 到多张卡上
        logits = model(input_ids, lengths, src_mask)  # [B, num_classes]
        loss = crit(logits, labels)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        preds = logits.argmax(dim=-1)
        total += loss.item() * input_ids.size(0)
        correct += (preds == labels).sum().item()
        n += input_ids.size(0)

    return total / n, correct / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    crit = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids, lengths, labels = [x.to(device) for x in batch]
        src_mask = make_pad_mask(lengths, max_len=input_ids.size(1), device=device)

        logits = model(input_ids, lengths, src_mask)  # [B, num_classes]
        loss = crit(logits, labels)

        preds = logits.argmax(dim=-1)
        total += loss.item() * input_ids.size(0)
        correct += (preds == labels).sum().item()
        n += input_ids.size(0)

    return total / n, correct / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dim', type=int, default=128)
    p.add_argument('--ffn_dim', type=int, default=512)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--save_dir', type=str, default='results/agnews_encoder')
    p.add_argument('--model_name', type=str, default='bert-base-uncased')
    args = p.parse_args()

    cfg = Config(**vars(args))
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Data
    ds = load_dataset('ag_news')
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_loader = DataLoader(ds['train'], batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tok, cfg.max_len))
    test_loader = DataLoader(ds['test'], batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tok, cfg.max_len))
    # Model
    num_classes = 4
    vocab_size = tok.vocab_size

    # 1. 检测设备并打印
    if torch.cuda.is_available():
        # 只用第 1 张卡（足够做作业了）
        torch.cuda.set_device(1)
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"[Device] 检测到 {n_gpu} 张 GPU，将使用第 1 张卡训练")
        print(f"  - GPU 1: {torch.cuda.get_device_name(1)}")
    else:
        device = torch.device("cpu")
        print("[Device] 未检测到 GPU，使用 CPU 训练")

    # 2. 构建模型（单卡）
    model = EncoderClassifier(
        vocab_size,
        num_classes,
        d_model=cfg.model_dim,
        num_layers=cfg.layers,
        num_heads=cfg.heads,
        d_ff=cfg.ffn_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)


    history = []
    best_acc = 0.0
    for epoch in range(1, cfg.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optim, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        rec = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
               'val_loss': val_loss, 'val_acc': val_acc}
        history.append(rec)
        print(rec)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'best.pt'))
        # plot
        plot_curve(history, 'epoch', 'train_loss', os.path.join(cfg.save_dir, 'train_loss.png'), title='Train Loss')
        plot_curve(history, 'epoch', 'val_loss', os.path.join(cfg.save_dir, 'val_loss.png'), title='Val Loss')

    save_jsonl(history, os.path.join(cfg.save_dir, 'metrics.jsonl'))
    print("Done. Best val_acc:", best_acc)

if __name__ == '__main__':
    main()
