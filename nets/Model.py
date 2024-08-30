import math
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from config import TSConfig


class MultiheadSelfAttention(torch.nn.Module):
    '''
    key pad mask shape must be 8,512,512
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

        Q: torch.Tensor = self.LQ(query).view(B, S, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)
        K: torch.Tensor = self.LK(key).view(B, S, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)
        V: torch.Tensor = self.LV(value).view(B, S, self.attention_head_n, E // self.attention_head_n).transpose(1, 2)

        energy = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embd_dim)

        if mask is not None:
            energy=energy.masked_fill(mask == 0, float('-inf'))
        softmax_energy = F.softmax(energy, -1)
        result = softmax_energy @ V
        result = result.transpose(1, 2).contiguous().view(B, S, E)
        return result


if __name__ == "__main__":
    config = TSConfig()
    dummy = torch.randn(size=(8, 512, 128))
    mask = torch.ones(size=(8, 512, 512))
    trg_mask = torch.tril(torch.ones((512, 512))).bool().unsqueeze(0)
    att = MultiheadSelfAttention(config)
    a = att(dummy, dummy, dummy, mask)
