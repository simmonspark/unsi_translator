import math
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from nets.config import TSConfig


class MultiheadSelfAttention(torch.nn.Module):
    '''
    key pad mask shape must be B,512,512
    trg mask shape must be 1,512,512
    '''

    def __init__(self, config):
        super().__init__()
        self.attention_head_n = config.attention_head_n
        self.embd_dim = config.embd_dim
        self.attention_drop = nn.Dropout(0.1)

        self.LQ = nn.Linear(self.embd_dim, self.embd_dim)
        self.LK = nn.Linear(self.embd_dim, self.embd_dim)
        self.LV = nn.Linear(self.embd_dim, self.embd_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):

        if mask is not None:
            assert len(mask.shape) == 3

            mask = mask.unsqueeze(dim=1)

        B, S, E = query.shape

        Q: torch.Tensor = self.LQ(query).view(B, -1, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)
        K: torch.Tensor = self.LK(key).view(B, -1, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)
        V: torch.Tensor = self.LV(value).view(B, -1, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)

        energy = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embd_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == False, float('-inf'))

        softmax_energy = F.softmax(energy, -1)
        result = softmax_energy @ V
        result = result.transpose(1, 2).contiguous().view(B, S, E)
        return result


class PosAndWordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd_layer = nn.Embedding(self.config.vocab_size, self.config.embd_dim)
        self.pos_embd = nn.Embedding(self.config.block_size, self.config.embd_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        b, t = x.size()
        x = self.embd_layer(x)

        device = x.device

        pos_raw_ids = torch.arange(0, t, dtype=torch.long, device=device)

        pos = self.pos_embd(pos_raw_ids)

        x = self.drop(x + pos)
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_dim, 2 * config.embd_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(2 * config.embd_dim, config.embd_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = MLP(config)
        self.drop = nn.Dropout(0.1)
        self.att = MultiheadSelfAttention(config)
        self.norm = nn.LayerNorm(config.embd_dim, bias=False)

    def forward(self, x, attention_mask=None):
        x = self.att(x, x, x, attention_mask)
        res = x
        x = self.head(x)
        x = res + x
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ebd = PosAndWordEmbedding(config)
        self.block_list = nn.ModuleList([TransformerBlock(config) for _ in range(config.encoder_layer_n)])

    def forward(self, x, attention_mask=None):
        x = self.ebd(x)
        for block in self.block_list:
            x = block(x, attention_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.att = MultiheadSelfAttention(config)
        self.transformerblock = nn.ModuleDict(dict(
            mhe=MultiheadSelfAttention(config),
            mlp=MLP(config),
            ln1=nn.LayerNorm(config.embd_dim, eps=1e-6),
            ln2=nn.LayerNorm(config.embd_dim, eps=1e-6),
            do=nn.Dropout(0.1),
        ))

    def forward(self, encoder_out, target, attn_mask, trg_mask):
        residual = target

        decoder_attention = self.att(target, target, target, trg_mask)

        decoder_attention = self.transformerblock.do(decoder_attention + residual)
        decoder_attention = self.transformerblock.ln1(decoder_attention)

        mha_output = self.transformerblock.mhe(decoder_attention, encoder_out, encoder_out, attn_mask)

        mha_output = self.transformerblock.do(mha_output + decoder_attention)
        mha_output = self.transformerblock.ln2(mha_output)

        res = mha_output
        decoder_attention_output = self.transformerblock.mlp(mha_output)
        decoder_attention_output = self.transformerblock.do(decoder_attention_output + res)
        decoder_attention_output = self.transformerblock.ln2(decoder_attention_output)

        return decoder_attention_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.posembd = PosAndWordEmbedding(config)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layer_n)])
        self.drop = nn.Dropout(0.1)

    def forward(self, target, encoder_out, attn_mask, trg_mask):
        target = self.drop(self.posembd(target))
        for block in self.blocks:
            target = block(encoder_out, target, attn_mask, trg_mask)
        return target


class TS(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.fc_out = nn.Linear(config.embd_dim, config.vocab_size)

    def make_trg_mask(self, seq):
        S = seq.shape[1]
        trg_mask = torch.tril(torch.ones((S, S), device=seq.device)).bool().unsqueeze(0)
        return trg_mask

    def forward(self, source, target, attention_mask=None):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source, attention_mask)
        outputs = self.decoder(target, enc_out, attention_mask, trg_mask)
        output = self.fc_out(outputs)

        return output


if __name__ == "__main__":
    config = TSConfig()
    dummy = torch.randint(10, 100, size=(8, 512))
    target = torch.randint(10,100,size = (8,2))
    mask = torch.ones(size=(8, 512)).unsqueeze(1)
    trg_mask = torch.tril(torch.ones((512, 512))).bool().unsqueeze(0)
    M = TS(config)
    a = M(dummy,target, mask)
