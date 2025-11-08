# src/train_lm_agnews.py
import os, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .models import DecoderLM
from .utils import set_seed, save_jsonl, plot_curve, subsequent_mask


def build_blocks_from_texts(texts, tokenizer, max_len):
    """
    把很多句子拼成一个长文本，然后切成若干长度为 max_len+1 的 block。
    每个 block 前 max_len 个 token 做输入，后 max_len 个 token 做目标。
    """
    # 拼成一个大字符串，中间用换行隔开
    big_text = "\n".join(texts)
    ids = tokenizer(big_text, return_tensors="pt").input_ids  # [1, N]
    ids = ids[0]  # [N]

    blocks = []
    step = max_len  # 可以用滑窗，这里简单用 non-overlap 步长= max_len
    for i in range(0, ids.size(0) - max_len - 1, step):
        block = ids[i : i + max_len + 1]  # 长度 max_len+1
        blocks.append({"ids": block})
    return blocks


def collate_autoregressive(batch, device):
    """
    batch: [{"ids": LongTensor[L+1]}, ...]
    返回:
      X: [B, L]  输入序列
      Y: [B, L]  目标序列(shifted)
    """
    X = torch.stack([b["ids"][:-1] for b in batch], dim=0)
    Y = torch.stack([b["ids"][1:] for b in batch], dim=0)
    return X.to(device), Y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="results/lm_agnews")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 1. 加载 AG News 文本数据 ==========
    # 离线也没问题：你现在已经有 ag_news 的缓存
    ds = load_dataset("ag_news")
    train_texts = [ex["text"] for ex in ds["train"]]
    valid_texts = [ex["text"] for ex in ds["test"]]

    # ========== 2. 加载 tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    # GPT2 默认没有 pad_token，这里把 pad_token 对齐到 eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 3. 文本 -> token blocks ==========
    print(f"[Data] build train blocks from {len(train_texts)} texts")
    train_blocks = build_blocks_from_texts(train_texts, tokenizer, args.max_len)
    print(f"[Data] train blocks: {len(train_blocks)}")

    print(f"[Data] build valid blocks from {len(valid_texts)} texts")
    valid_blocks = build_blocks_from_texts(valid_texts, tokenizer, args.max_len)
    print(f"[Data] valid blocks: {len(valid_blocks)}")

    train_loader = DataLoader(
        train_blocks,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_autoregressive(b, device),
    )
    valid_loader = DataLoader(
        valid_blocks,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_autoregressive(b, device),
    )

    # ========== 4. 构建 Decoder-only 模型 ==========
    model = DecoderLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.model_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        d_ff=args.ffn_dim,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ---------- 训练 ----------
        model.train()
        ep_loss, n_tok = 0.0, 0
        for X, Y in tqdm(train_loader, desc=f"train[{epoch}]"):
            L = X.size(1)
            # causal mask: [1, L, L]，True 表示要 mask
            causal = subsequent_mask(L, device=device)

            logits = model(X, tgt_mask=causal)  # [B, L, V]
            loss = crit(logits.view(-1, logits.size(-1)), Y.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_loss += loss.item() * X.numel()
            n_tok += X.numel()
            global_step += 1

        train_loss = ep_loss / n_tok

        # ---------- 验证 ----------
        model.eval()
        ep_loss, n_tok = 0.0, 0
        with torch.no_grad():
            for X, Y in tqdm(valid_loader, desc=f"valid[{epoch}]"):
                L = X.size(1)
                causal = subsequent_mask(L, device=device)
                logits = model(X, tgt_mask=causal)
                loss = crit(logits.view(-1, logits.size(-1)), Y.view(-1))
                ep_loss += loss.item() * X.numel()
                n_tok += X.numel()
        val_loss = ep_loss / n_tok
        val_ppl = math.exp(val_loss)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        }
        history.append(rec)
        print(rec)

        # 画曲线
        plot_curve(history, "epoch", "train_loss", os.path.join(args.save_dir, "train_loss.png"), title="Train Loss")
        plot_curve(history, "epoch", "val_loss", os.path.join(args.save_dir, "val_loss.png"), title="Val Loss")

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model_last.pt"))

    save_jsonl(history, os.path.join(args.save_dir, "metrics.jsonl"))
    print("Done. Last val_ppl:", history[-1]["val_ppl"])


if __name__ == "__main__":
    main()
