from collections import Counter
from pathlib import Path

from .tokenizer import GraphiteTokenizer


DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


BYTE_VOCAB_SIZE = 256

# A pair must occur at least this many times before it can become
# a learned BPE merge. This prevents tiny datasets from collapsing
# into giant one-off tokens.
MIN_PAIR_FREQUENCY = 2


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".lua",
    ".luau",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".java",
    ".rs",
    ".go",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".cmake",
    ".sh",
    ".ps1",
}


def byte_token(value: int) -> str:
    """Convert a byte value into a tokenizer symbol."""

    return f"<byte:{value}>"


def load_corpus(
    data_directory: str | Path,
) -> str:
    """Load all supported files into one training corpus."""

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_directory}"
        )

    texts = []

    for path in sorted(
        data_directory.rglob("*")
    ):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        if text:
            texts.append(text)

    if not texts:
        raise ValueError(
            f"No supported training files found in: "
            f"{data_directory}"
        )

    return "\n\n".join(texts)


def build_initial_vocabulary(
    special_tokens: dict[str, int],
) -> dict[str, int]:
    """
    Build the initial vocabulary.

    Special tokens are followed by all 256 possible byte tokens.
    """

    vocabulary = dict(special_tokens)

    used_ids = set(
        vocabulary.values()
    )

    next_id = max(
        used_ids,
        default=-1,
    ) + 1

    for value in range(
        BYTE_VOCAB_SIZE
    ):
        token = byte_token(value)

        if token in vocabulary:
            continue

        while next_id in used_ids:
            next_id += 1

        vocabulary[token] = next_id
        used_ids.add(next_id)

        next_id += 1

    return vocabulary


def encode_corpus(
    corpus: str,
) -> list[str]:
    """
    Convert the complete corpus into UTF-8 byte symbols.

    Spaces, newlines, indentation, punctuation, and other formatting
    remain part of the training sequence.
    """

    encoded = corpus.encode(
        "utf-8"
    )

    return [
        byte_token(value)
        for value in encoded
    ]


def count_pairs(
    symbols: list[str],
) -> Counter[tuple[str, str]]:
    """Count adjacent symbol pairs across the corpus."""

    pair_counts = Counter()

    for index in range(
        len(symbols) - 1
    ):
        pair = (
            symbols[index],
            symbols[index + 1],
        )

        pair_counts[pair] += 1

    return pair_counts


def merge_pair(
    symbols: list[str],
    pair: tuple[str, str],
) -> list[str]:
    """Merge every occurrence of a pair in the corpus."""

    left, right = pair
    merged_symbol = left + right

    merged = []
    index = 0

    while index < len(symbols):
        if (
            index < len(symbols) - 1
            and symbols[index] == left
            and symbols[index + 1] == right
        ):
            merged.append(
                merged_symbol
            )

            index += 2

        else:
            merged.append(
                symbols[index]
            )

            index += 1

    return merged


def train_bpe(
    corpus: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
    min_pair_frequency: int = MIN_PAIR_FREQUENCY,
) -> GraphiteTokenizer:
    """
    Train a byte-level BPE tokenizer over the complete corpus.

    Pairs that occur fewer than min_pair_frequency times are not
    merged. This prevents tiny datasets from creating giant tokens
    that merely memorize unique sections of the corpus.
    """

    special_tokens = (
        special_tokens.copy()
        if special_tokens is not None
        else DEFAULT_SPECIAL_TOKENS.copy()
    )

    if min_pair_frequency < 1:
        raise ValueError(
            "min_pair_frequency must be at least 1."
        )

    minimum_vocab_size = (
        len(special_tokens)
        + BYTE_VOCAB_SIZE
    )

    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size must be at least "
            f"{minimum_vocab_size} for byte-level BPE."
        )

    vocabulary = build_initial_vocabulary(
        special_tokens
    )

    symbols = encode_corpus(
        corpus
    )

    if not symbols:
        raise ValueError(
            "Corpus contains no usable UTF-8 data."
        )

    merges: list[tuple[str, str]] = []

    while len(vocabulary) < vocab_size:
        pair_counts = count_pairs(
            symbols
        )

        if not pair_counts:
            break

        eligible_pairs = {
            pair: count
            for pair, count in pair_counts.items()
            if count >= min_pair_frequency
        }

        if not eligible_pairs:
            break

        best_pair = max(
            eligible_pairs,
            key=lambda pair: (
                eligible_pairs[pair],
                pair,
            ),
        )

        merged_symbol = (
            best_pair[0]
            + best_pair[1]
        )

        if merged_symbol in vocabulary:
            symbols = merge_pair(
                symbols,
                best_pair,
            )

            continue

        vocabulary[
            merged_symbol
        ] = len(vocabulary)

        merges.append(
            best_pair
        )

        symbols = merge_pair(
            symbols,
            best_pair
        )

    return GraphiteTokenizer(
        vocabulary=vocabulary,
        merges=merges,
        special_tokens=special_tokens,
    )


def train_tokenizer(
    data_directory: str | Path,
    output_path: str | Path,
    vocab_size: int = 32000,
    min_pair_frequency: int = MIN_PAIR_FREQUENCY,
) -> GraphiteTokenizer:
    """Train and save the Graphite tokenizer."""

    corpus = load_corpus(
        data_directory
    )

    print(
        f"Corpus characters: "
        f"{len(corpus):,}"
    )

    print(
        f"Corpus bytes: "
        f"{len(corpus.encode('utf-8')):,}"
    )

    print(
        f"Minimum pair frequency: "
        f"{min_pair_frequency}"
    )

    tokenizer = train_bpe(
        corpus=corpus,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
    )

    tokenizer.save(
        output_path
    )

    return tokenizer


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    data_directory = (
        project_root
        / "data"
        / "datasets"
    )

    output_path = (
        project_root
        / "tokenizer"
        / "files"
        / "tokenizer.json"
    )

    tokenizer = train_tokenizer(
        data_directory=data_directory,
        output_path=output_path,
        vocab_size=32000,
        min_pair_frequency=MIN_PAIR_FREQUENCY,
    )

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Learned merges: "
        f"{len(tokenizer.merges)}"
    )

    print(
        f"Tokenizer written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()