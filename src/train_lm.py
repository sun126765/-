import os, math, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from .models import DecoderLM
from .utils import set_seed, save_jsonl, plot_curve, subsequent_mask

def load_text_dataset(name='tiny_shakespeare'):
    if name == 'tiny_shakespeare':
        ds = load_dataset('tiny_shakespeare')
        text = ds['train'][0]['text']
        return {'train': [{'text': text[:800000]}], 'valid': [{'text': text[800000:900000]}]}
    elif name == 'wikitext-2':
        return load_dataset('wikitext', 'wikitext-2-raw-v1')
    else:
        raise ValueError("Unsupported dataset: " + name)

def encode_lines(tokenizer, texts, max_len):
    toks = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt', return_attention_mask=False)
    return toks['input_ids']

def make_batches(token_ids, block_size, step):
    # slice long sequence into overlapping blocks
    blocks = []
    for i in range(0, token_ids.size(1) - block_size - 1, step):
        blocks.append(token_ids[:, i:i+block_size+1])  # +1 for next token
    return torch.cat(blocks, dim=0) if blocks else token_ids

def collate_autoregressive(batch, block_size, device):
    X = torch.stack([b['ids'][:-1] for b in batch])  # [B, L]
    Y = torch.stack([b['ids'][1:] for b in batch])   # [B, L]
    return X.to(device), Y.to(device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dim', type=int, default=128)
    p.add_argument('--ffn_dim', type=int, default=512)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', type=str, default='tiny_shakespeare', choices=['tiny_shakespeare','wikitext-2'])
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--save_dir', type=str, default='results/lm_decoder')
    p.add_argument('--tokenizer_name', type=str, default='gpt2')
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    raw = load_text_dataset(args.dataset)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # For tiny Shakespeare, encode big text into tokens then make blocks
    if args.dataset == 'tiny_shakespeare':
        text = raw['train'][0]['text']
        ids = tok(text, return_tensors='pt').input_ids
        # build blocks of length max_len+1
        blocks = []
        step = args.max_len
        for i in range(0, ids.size(1) - args.max_len - 1, step):
            blk = ids[:, i:i+args.max_len+1].squeeze(0)
            blocks.append({'ids': blk})
        train_data = blocks[: max(1, int(len(blocks)*0.95))]
        valid_data = blocks[max(1, int(len(blocks)*0.95)) : ]
    else:
        # build from multiple lines
        train_texts = [x['text'] for x in raw['train'] if x['text'].strip()]
        valid_texts = [x['text'] for x in raw['validation'] if x['text'].strip()]
        train_ids = tok(train_texts, padding=True, truncation=True, max_length=args.max_len+1, return_tensors='pt').input_ids
        valid_ids = tok(valid_texts, padding=True, truncation=True, max_length=args.max_len+1, return_tensors='pt').input_ids
        train_data = [{'ids': row} for row in train_ids]
        valid_data = [{'ids': row} for row in valid_ids]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_autoregressive(b, args.max_len, device))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_autoregressive(b, args.max_len, device))

    # Model
    model = DecoderLM(vocab_size=tok.vocab_size, d_model=args.model_dim, num_layers=args.layers,
                      num_heads=args.heads, d_ff=args.ffn_dim, dropout=args.dropout, max_len=args.max_len).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss, n_tok = 0.0, 0
        for X, Y in tqdm(train_loader, desc=f"train[{epoch}]"):
            L = X.size(1)
            causal = subsequent_mask(L, device=device)  # [1,L,L]
            logits = model(X, tgt_mask=causal)  # [B, L, V]
            loss = crit(logits.view(-1, logits.size(-1)), Y.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_loss += loss.item() * X.numel()
            n_tok += X.numel()
        train_loss = ep_loss / n_tok
        # eval
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
        rec = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_ppl': math.exp(val_loss)}
        history.append(rec)
        print(rec)
        plot_curve(history, 'epoch', 'train_loss', os.path.join(args.save_dir, 'train_loss.png'), title='Train Loss')
        plot_curve(history, 'epoch', 'val_loss', os.path.join(args.save_dir, 'val_loss.png'), title='Val Loss')
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_last.pt'))

    save_jsonl(history, os.path.join(args.save_dir, 'metrics.jsonl'))
    print("Done. Last val_ppl:", history[-1]['val_ppl'])

if __name__ == '__main__':
    main()
