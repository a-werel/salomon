"""The single LSTM architecture used by the inference image and checkpoints."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AstLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(output, batch_first=True)
        pooled = states.sum(dim=1) / lengths.to(states.device).unsqueeze(1)
        return self.classifier(pooled)
