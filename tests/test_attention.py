"""Test the python function from src."""

import sys

sys.path.insert(0, "./src/")

import pytest
import torch

from src.attention_model import dot_product_attention


@pytest.mark.parametrize("batch", [16, 32])
@pytest.mark.parametrize("heads", [2, 4, 12])
@pytest.mark.parametrize("emb", [256, 512])
@pytest.mark.parametrize("dv", [64, 128])
@pytest.mark.parametrize("dk", [64, 128])
def test_attention(batch: int, heads: int, emb: int, dv: int, dk: int) -> None:
    """Compare to torch.nn.functional.scaled_dot_product_attention."""
    q = torch.randn([batch, heads, emb, dk]).type(torch.float64)
    k = torch.randn([batch, heads, emb, dk]).type(torch.float64)
    v = torch.randn([batch, heads, emb, dv]).type(torch.float64)

    torch_attn = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )
    my_attn = dot_product_attention(q, k, v)

    assert torch.allclose(torch_attn, my_attn, atol=1e-7)
