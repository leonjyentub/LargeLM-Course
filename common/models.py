from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FFNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 128, max_len: int = 7, num_classes: int = 2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(emb_dim * max_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        flat = emb.reshape(emb.size(0), -1)
        return self.net(flat)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 48, hidden_dim: int = 128, num_classes: int = 2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 48, hidden_dim: int = 128, num_classes: int = 2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        return self.fc(h_n[-1])


class LSTMMultiLabelClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 48, hidden_dim: int = 128, positions: int = 4, n_digits: int = 10):
        super().__init__()
        self.positions = positions
        self.n_digits = n_digits
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, positions * n_digits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        logits = self.fc(h_n[-1])
        return logits.reshape(x.size(0), self.positions, self.n_digits)


class SelfAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_classes: int = 2000,
        max_len: int = 7,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len=max_len)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_mask = x.eq(0)
        emb = self.pos(self.embedding(x))
        attn_out, _ = self.attn(emb, emb, emb, key_padding_mask=pad_mask)
        h = self.norm1(emb + attn_out)
        h2 = self.ff(h)
        h = self.norm2(h + h2)

        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_emb(src)
        _, (h, c) = self.encoder(src_emb)

        tgt_emb = self.tgt_emb(tgt_in)
        dec_out, _ = self.decoder(tgt_emb, (h, c))
        return self.fc(dec_out)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        src_emb = self.src_emb(src)
        _, (h, c) = self.encoder(src_emb)

        bsz = src.size(0)
        cur = torch.full((bsz, 1), sos_id, dtype=torch.long, device=src.device)
        outputs = []

        for _ in range(max_len):
            emb = self.tgt_emb(cur[:, -1:])
            dec_out, (h, c) = self.decoder(emb, (h, c))
            logits = self.fc(dec_out[:, -1])
            nxt = logits.argmax(dim=-1, keepdim=True)
            outputs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)

        return torch.cat(outputs, dim=1)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        src_max_len: int = 7,
        tgt_max_len: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.src_pos = PositionalEncoding(d_model, src_max_len)
        self.tgt_pos = PositionalEncoding(d_model, tgt_max_len + 1)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_pad = src.eq(0)
        tgt_pad = tgt_in.eq(0)

        src_emb = self.src_pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.tgt_pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
            tgt_mask=tgt_mask,
        )
        return self.fc(out)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        bsz = src.size(0)
        ys = torch.full((bsz, 1), sos_id, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            logits = self.forward(src, ys)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)

        return ys[:, 1:]
