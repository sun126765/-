import os, math, argparse, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sacrebleu import corpus_bleu
from .models import Seq2SeqTransformer
from .utils import set_seed, save_jsonl, plot_curve, subsequent_mask, make_pad_mask

def collate_mt(batch, tok_src, tok_tgt, src_lang, tgt_lang, max_len, device):
    src = [b['translation'][src_lang] for b in batch]
    tgt = [b['translation'][tgt_lang] for b in batch]
    src_out = tok_src(src, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    tgt_out = tok_tgt(tgt, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    return (src_out['input_ids'].to(device), src_out['attention_mask'].sum(1).to(device),
            tgt_out['input_ids'].to(device), tgt_out['attention_mask'].sum(1).to(device))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dim', type=int, default=256)
    p.add_argument('--ffn_dim', type=int, default=1024)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--enc_layers', type=int, default=4)
    p.add_argument('--dec_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--src_lang', type=str, default='en')
    p.add_argument('--tgt_lang', type=str, default='de')
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--limit_train', type=int, default=10000)
    p.add_argument('--save_dir', type=str, default='results/iwslt_en_de')
    p.add_argument('--tok_src', type=str, default='bert-base-uncased')
    p.add_argument('--tok_tgt', type=str, default='bert-base-german-cased')
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    ds = load_dataset('iwslt2017', f'{args.src_lang}-{args.tgt_lang}')
    if args.limit_train and args.limit_train > 0:
        ds['train'] = ds['train'].select(range(min(args.limit_train, len(ds['train']))))
    tok_src = AutoTokenizer.from_pretrained(args.tok_src, use_fast=True)
    tok_tgt = AutoTokenizer.from_pretrained(args.tok_tgt, use_fast=True)
    if tok_tgt.pad_token_id is None: tok_tgt.pad_token = tok_tgt.eos_token
    train_loader = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_mt(b, tok_src, tok_tgt, args.src_lang, args.tgt_lang, args.max_len, device))
    valid_loader = DataLoader(ds['validation'], batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_mt(b, tok_src, tok_tgt, args.src_lang, args.tgt_lang, args.max_len, device))

    model = Seq2SeqTransformer(tok_src.vocab_size, tok_tgt.vocab_size, d_model=args.model_dim,
                               num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
                               num_heads=args.heads, d_ff=args.ffn_dim, dropout=args.dropout, max_len=args.max_len).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(ignore_index=tok_tgt.pad_token_id)

    history = []
    best_bleu = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss, n_tok = 0.0, 0
        for src_ids, src_len, tgt_ids, tgt_len in tqdm(train_loader, desc=f"train[{epoch}]"):
            # Prepare masks
            src_mask = make_pad_mask(src_len, max_len=src_ids.size(1), device=device)  # [B,1,1,Sl]
            L = tgt_ids.size(1) - 1
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]
            tgt_causal = subsequent_mask(L, device=device)  # [1,L,L]
            # cross-attn mask: pad mask on encoder keys
            memory_mask = make_pad_mask(src_len, max_len=src_ids.size(1), device=device).expand(-1, 1, L, -1)
            logits = model(src_ids, tgt_in, src_mask=src_mask, tgt_mask=tgt_causal, memory_mask=memory_mask)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_loss += loss.item() * tgt_out.numel()
            n_tok += tgt_out.numel()
        train_loss = ep_loss / n_tok

        # Eval BLEU
        model.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for src_ids, src_len, tgt_ids, tgt_len in tqdm(valid_loader, desc=f"valid[{epoch}]"):
                src_mask = make_pad_mask(src_len, max_len=src_ids.size(1), device=device)
                bos_id = tok_tgt.bos_token_id or tok_tgt.cls_token_id or tok_tgt.sep_token_id or tok_tgt.eos_token_id
                eos_id = tok_tgt.eos_token_id or tok_tgt.sep_token_id
                if bos_id is None: bos_id = tok_tgt.pad_token_id
                if eos_id is None: eos_id = tok_tgt.pad_token_id
                out_ids = model.greedy_generate(src_ids, bos_id, eos_id, max_len=args.max_len, src_mask=src_mask)
                # decode
                for hyp, ref in zip(out_ids.tolist(), tgt_ids.tolist()):
                    hyps.append(tok_tgt.decode(hyp, skip_special_tokens=True))
                    refs.append([tok_tgt.decode(ref, skip_special_tokens=True)])
        bleu = corpus_bleu(hyps, refs).score
        history.append({'epoch': epoch, 'train_loss': train_loss, 'bleu': bleu})
        print(history[-1])
        plot_curve(history, 'epoch', 'train_loss', os.path.join(args.save_dir, 'train_loss.png'), title='Train Loss')
        plot_curve(history, 'epoch', 'bleu', os.path.join(args.save_dir, 'bleu.png'), title='BLEU')
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))

    save_jsonl(history, os.path.join(args.save_dir, 'metrics.jsonl'))
    print("Done. Best BLEU:", best_bleu)

if __name__ == '__main__':
    main()
