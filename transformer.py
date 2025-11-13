# Attention is all you need
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len].to(x.device)
        return x

# Multi-Head Attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):

        B = query.size(0)

        Q = self.Wq(query) 
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)  

        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        out = self.Wo(context)
        return out


# Feed-Forward Network

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# Encoder Layer


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


# Decoder Layer

class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x


# Encoder


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# Decoder

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.embedding(tgt)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)

        return x


# Encoder + Decoder

class Transformer(nn.Module):

    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, num_layers=6, num_heads=8, d_ff=2048,
                 max_len=5000):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len)

        self.output_layer = nn.Linear(d_model, tgt_vocab)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, L = tgt.shape
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        subseq_mask = torch.tril(torch.ones((L, L), device=tgt.device)).bool()
        subseq_mask = subseq_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask & subseq_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, src_mask)

        logits = self.output_layer(out)
        return logits


# Example

if __name__ == "__main__":
    src_vocab = tgt_vocab = 10000
    model = Transformer(src_vocab, tgt_vocab)

    batch_size = 2
    src_seq = torch.randint(1, src_vocab, (batch_size, 20))  # source sentence
    tgt_seq = torch.randint(1, tgt_vocab, (batch_size, 20))  # target sentence

    out = model(src_seq, tgt_seq)
    print("Output logits:", out.shape)  # (batch, seq_len, vocab)
