import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with rotary positional embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        context_length: int,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.query_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.key_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.value_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.rotary_embedding = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=context_length,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query = self.rotary_embedding(
            query,
            position_ids,
        )

        key = self.rotary_embedding(
            key,
            position_ids,
        )

        sequence_length = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        attention_output = self._merge_heads(
            attention_output
        )

        return self.output_projection(attention_output)