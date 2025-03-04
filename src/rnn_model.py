"""Implement a recurrent LSTM-based language model.

Reference
- https://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """An LSTM-Cell."""

    def __init__(self, vocab_size: int, hidden_size: int):
        """Set up the LSTM-Network."""
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_size)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """Compute the forward pass of the LSTM-Net.

        Args:
            x (torch.Tensor): Input of shape [batch, time, vocab_size].
        """
        batch_size = x.shape[0]
        h_old = torch.zeros([batch_size, self.hidden_size]).to(x.device)
        c_old = torch.zeros([batch_size, self.hidden_size]).to(x.device)

        out_lst = []
        time = x.shape[1]
        for t in range(time):
            h_new, c_new = self.lstm_cell(x[:, t, :], (h_old, c_old))
            y_new = self.out_proj(h_new)
            out_lst.append(y_new)
            h_old = h_new
            c_old = c_new

        return torch.stack(out_lst, 1)

    @torch.no_grad()
    def sample(self, x, length):
        """Sample from the trained network."""
        batch_size = x.shape[0]
        h_old = torch.zeros([batch_size, self.hidden_size]).to(x.device)
        c_old = torch.zeros([batch_size, self.hidden_size]).to(x.device)

        out_lst = []
        embedding_size = x.shape[-1]
        x = x[:, 0, :]
        for _ in range(length):
            h_new, c_new = self.lstm_cell(x, (h_old, c_old))
            y_new = self.out_proj(h_new)
            out_lst.append(y_new)
            h_old = h_new
            c_old = c_new
            x = torch.nn.functional.one_hot(
                torch.argmax(y_new, -1), embedding_size
            ).type(torch.float32)

        return torch.stack(out_lst, 1)
