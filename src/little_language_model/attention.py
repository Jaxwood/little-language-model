import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim, bias=bias)
        self.key = nn.Linear(input_dim, output_dim, bias=bias)
        self.value = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)
        scores = queries @ keys.T
        weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)
        context = weights @ values
        return context
