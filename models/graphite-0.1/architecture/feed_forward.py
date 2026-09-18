import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Transformer feed-forward network.

    Expands the hidden representation into an intermediate dimension,
    applies a nonlinear activation, then projects it back to the
    model hidden size.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ):
        super().__init__()

        self.up_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_projection = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        if activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_projection(x)
        x = self.activation(x)
        x = self.down_projection(x)

        return x