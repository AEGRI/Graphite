import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Applies position-dependent rotations to query and key representations.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        positions = torch.arange(max_seq_len, dtype=torch.float32)

        frequencies = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        angles = torch.outer(positions, frequencies)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply rotary position encoding.

        Expected x shape:
            [batch, heads, sequence, head_dim]
        """

        if position_ids is None:
            position_ids = torch.arange(
                x.size(-2),
                device=x.device,
            )

        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)