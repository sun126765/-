#!/usr/bin/env bash
set -e

TASK=${1:-cls}

case "$TASK" in
  cls)
    echo "[Run] Encoder-only text classification (AG News)"
    python -m src.train_cls       --model_dim 128 --ffn_dim 512 --heads 4 --layers 2 --dropout 0.1       --batch_size 64 --lr 3e-4 --epochs 5 --seed 42       --max_len 256 --save_dir results/agnews_encoder
    ;;
  lm)
    echo "[Run] Decoder-only language modeling (Tiny Shakespeare)"
    python -m src.train_lm       --model_dim 128 --ffn_dim 512 --heads 4 --layers 4 --dropout 0.1       --batch_size 64 --lr 3e-4 --epochs 5 --seed 42       --dataset tiny_shakespeare --max_len 256 --save_dir results/lm_decoder
    ;;
  seq2seq)
    echo "[Run] Encoder-Decoder translation (IWSLT2017 EN↔DE)"
    python -m src.train_seq2seq       --model_dim 256 --ffn_dim 1024 --heads 4 --enc_layers 4 --dec_layers 4 --dropout 0.1       --batch_size 64 --lr 3e-4 --epochs 5 --seed 42       --src_lang en --tgt_lang de --max_len 128 --limit_train 10000       --save_dir results/iwslt_en_de
    ;;
  *)
    echo "Unknown TASK=$TASK, use one of: cls | lm | seq2seq"; exit 1;;
esac
