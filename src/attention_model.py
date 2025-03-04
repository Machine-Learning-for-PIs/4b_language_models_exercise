"""Minimal definition of a GPT-Language model.

References:
- [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
      Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin:
      Attention is All you Need. NIPS 2017: 5998-6008
      https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- [2] https://github.com/karpathy/nanoGPT/tree/master
"""

import torch
import torch.nn as nn
from torch.nn import functional as f


def dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    """Compute dot product attention.

    Args:
        q (torch.Tensor): The query tensor of shape [batch, heads, out_length, d_k].
        k (torch.Tensor): The key tensor of shape [batch, heads, out_length, d_k].
        v (torch.Tensor): The value-tensor of shape [batch, heads, out_length, d_v].

    Returns:
        torch.Tensor: The attention values of shape  [batch, heads, out_length, d_v]
    """
    # TODO implement multi head attention.
    # Use i.e. torch.transpose, torch.sqrt, torch.tril, torch.exp, torch.inf
    # as well as torch.nn.functional.softmax .
    attention_out = None
    return attention_out


class MultiHeadAttention(nn.Module):
    """Let's understand multi-head self attention."""

    def __init__(self, embedding_size: int, dropout_prob: float):
        """Initialize the attention module."""
        super().__init__()
        self.embedding_size = embedding_size
        self.embed_qkv = nn.Linear(embedding_size, 3 * embedding_size)
        self.back_proj = nn.Linear(embedding_size, embedding_size)

        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Run the multi-head attention forward pass."""
        queries, keys, values = self.embed_qkv(x).split(self.embedding_size, dim=2)
        attention_heads = dot_product_attention(queries, keys, values)
        self.attention_dropout(attention_heads)
        attention_projection = self.back_proj(attention_heads)
        self.output_dropout(attention_projection)
        return attention_projection


class FeedForwardNetwork(nn.Module):
    """A feedforward network.

    See also figure 1 of [1].
    """

    def __init__(self, embedding_size: int, dropout_prob: float):
        """Set up the feedforward network elements."""
        super().__init__()
        self.dense1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(embedding_size * 4, embedding_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Feedforward forward pass."""
        return self.dropout(self.dense2(self.act(self.dense1(x))))


class NetworkBlock(nn.Module):
    """A network block.

    See also figure 1 of [1].
    """

    def __init__(self, embedding_size: int, dropout_prob: float):
        """Set up the transformer network block."""
        super().__init__()
        self.attention = MultiHeadAttention(embedding_size, dropout_prob)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.feedforward = FeedForwardNetwork(embedding_size, dropout_prob)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        """Forward pass with residual connections."""
        x = x + self.attention(self.layer_norm(x))
        x = x + self.feedforward(self.layer_norm2(x))
        return x


class Transformer(nn.Module):
    """A minimal transformer network implementation.

    Reference implementation: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        block_size: int,
        dropout_prob: float,
        layer_number: int,
    ):
        """Instantiate a transformer."""
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(vocab_size, embedding_size),
                wpe=nn.Linear(block_size, embedding_size),
                drop=nn.Dropout(dropout_prob),
                h=nn.ModuleList(
                    [
                        NetworkBlock(embedding_size, dropout_prob)
                        for _ in range(layer_number)
                    ]
                ),
                ln_f=nn.LayerNorm(embedding_size),
            )
        )
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, idx):
        """Compute the transformer forward pass."""
        device = idx.device
        _, t, _ = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos = torch.nn.functional.one_hot(pos, self.block_size).type(torch.float32)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def sample(self, idx, max_new_tokens):
        """Take a conditioning sequence of indices idx and complete it.

        Predictions are fed back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities
            probs = f.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.nn.functional.one_hot(idx_next, self.vocab_size)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
