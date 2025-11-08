from typing import Optional, Tuple
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # 兼容 decoder-only 场景：如果没给 key/value，就退化成自注意力
        if key is None or value is None:
            key = value = query

        # query/key/value: [B, L, D]; mask shape can be broadcast to [B, h, Lq, Lk]
        B, Lq, _ = query.size()
        Lk = key.size(1)

        def shape(x):
            # [B, L, D] -> [B, h, L, d_k]
            return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q = shape(self.w_q(query))
        K = shape(self.w_k(key))
        V = shape(self.w_v(value))

        # [B, h, Lq, Lk]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask: True 表示要 mask，能 broadcast 到 [B, 1 or h, Lq, Lk]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # [B, h, Lq, d_k]
        x = torch.matmul(attn, V)
        # [B, Lq, D]
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(x)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, tgt_mask=None, memory_mask=None):
        # 1) decoder 自注意力（带 causal mask）
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))

        # 2) 只有在有 memory（encoder-decoder）时才做 cross-attn
        if memory is not None:
            x2 = self.cross_attn(x, memory, memory, memory_mask)
            x = self.norm2(x + self.dropout(x2))
        # 如果是 decoder-only（memory=None），这一步直接跳过，
        # 相当于「没有 cross attention」的纯 decoder-only Transformer

        # 3) FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None, tgt_mask=None, memory_mask=None):
        # 1) masked self-attention
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))

        # 2) 只有在 encoder-decoder 结构中，memory 才不是 None
        if memory is not None:
            x2 = self.cross_attn(x, memory, memory, memory_mask)
            x = self.norm2(x + self.dropout(x2))
        # 如果是 decoder-only（memory=None），这一步直接跳过，相当于“没有 cross-attn”

        # 3) FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.pe(self.embed(src))
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.pe(self.embed(tgt))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)

class EncoderClassifier(nn.Module):
    """Encoder-only：句子分类（支持长度信息的 masked mean 池化）"""
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        max_len=512,
        pool="mean",
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.pool = pool
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, lengths=None, src_mask=None):
        """
        src: [B, L]
        lengths: [B]  每个样本的有效长度（不含 padding）
        src_mask: [B, 1, 1, L]  padding mask（True 为被 mask）
        """
        enc = self.encoder(src, src_mask)  # [B, L, D]

        if lengths is not None:
            B, L, D = enc.shape
            device = enc.device
            # 构造 [B, L, 1] 的 mask，pad 位置为 0
            mask = (torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1))  # [B, L]
            mask = mask.float().unsqueeze(-1)  # [B, L, 1]
            summed = (enc * mask).sum(dim=1)   # [B, D]
            denom = lengths.clamp_min(1).unsqueeze(1).float()
            out = summed / denom               # masked mean
        elif self.pool == "mean":
            out = enc.mean(dim=1)
        else:
            out = enc[:, 0]  # [CLS]

        return self.fc(out)  # [B, num_classes]


class DecoderLM(nn.Module):
    """Decoder-only：自回归 LM"""
    def __init__(self, vocab_size, d_model=128, num_layers=4, num_heads=4, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt, tgt_mask=None):
        dec = self.decoder(tgt, memory=None, tgt_mask=tgt_mask, memory_mask=None)
        return self.lm_head(dec)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_encoder_layers=4, num_decoder_layers=4, num_heads=4, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_encoder_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_decoder_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        dec = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.fc_out(dec)

    @torch.no_grad()
    def greedy_generate(self, src, bos_id: int, eos_id: int, max_len: int, src_mask=None):
        device = src.device
        memory = self.encoder(src, src_mask)
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len-1):
            L = ys.size(1)
            causal = torch.triu(torch.ones(1, L, L, device=device), diagonal=1).bool()
            logits = self.forward(src, ys, src_mask, causal, None)  # [B, L, V]
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if (ys[:, -1] == eos_id).all():
                break
        return ys
