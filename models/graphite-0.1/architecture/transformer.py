import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """
    A single decoder-only Transformer block.

    Uses pre-normalization with RMSNorm, causal self-attention,
    rotary positional embeddings, and a feed-forward network.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        context_length: int,
        normalization_epsilon: float = 1e-5,
        activation: str = "silu",
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            hidden_size,
            eps=normalization_epsilon,
        )

        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            context_length=context_length,
        )

        self.feed_forward_norm = RMSNorm(
            hidden_size,
            eps=normalization_epsilon,
        )

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the transformer block.

        Args:
            x: Hidden states with shape
               [batch, sequence, hidden_size].
            position_ids: Optional position IDs for RoPE.

        Returns:
            Updated hidden states with the same shape as x.
        """

        x = x + self.attention(
            self.attention_norm(x),
            position_ids=position_ids,
        )

        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x