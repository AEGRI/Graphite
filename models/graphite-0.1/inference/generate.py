import torch

from .runtime import GraphiteRuntime


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the next token using temperature, top-k, and top-p."""

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1.")

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        threshold = values[..., -1, None]

        logits = logits.masked_fill(
            logits < threshold,
            torch.finfo(logits.dtype).min,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        remove_tokens = cumulative_probabilities > top_p

        # Keep the first token that crosses the threshold.
        remove_tokens[..., 1:] = remove_tokens[..., :-1].clone()
        remove_tokens[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_tokens,
            torch.finfo(sorted_logits.dtype).min,
        )

        logits = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )

        logits.scatter_(
            -1,
            sorted_indices,
            sorted_logits,
        )

    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )


@torch.no_grad()
def generate(
    runtime: GraphiteRuntime,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate token IDs autoregressively.

    Returns the original input IDs followed by generated IDs.
    Generation stops when all batch sequences emit EOS or the token
    limit is reached.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    if input_ids.dim() != 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence]."
        )

    if input_ids.size(1) == 0:
        raise ValueError("input_ids cannot have an empty sequence.")

    if do_sample and temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    context_length = runtime.model.context_length

    if context_length <= 0:
        raise ValueError("model context_length must be positive.")

    # Use a local generator so seeded generation is reproducible
    # without resetting PyTorch's global random state.
    generator = None

    if seed is not None:
        generator = torch.Generator(device=runtime.device)
        generator.manual_seed(seed)

    generated_ids = input_ids.to(runtime.device)

    # Read EOS from the loaded runtime tokenizer when available.
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(runtime, "tokenizer_instance", None)

    special_tokens = getattr(tokenizer, "special_tokens", {})
    eos_token_id = special_tokens.get("<eos>")

    finished = torch.zeros(
        generated_ids.size(0),
        dtype=torch.bool,
        device=runtime.device,
    )

    for _ in range(max_new_tokens):
        model_input = generated_ids[:, -context_length:]
        logits = runtime.forward(model_input)
        next_token_logits = logits[:, -1, :]

        if do_sample:
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
        else:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

        if eos_token_id is not None:
            # Preserve completed sequences while the remaining batch
            # continues generating.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=-1,
        )

        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)

            if finished.all():
                break

    return generated_ids
