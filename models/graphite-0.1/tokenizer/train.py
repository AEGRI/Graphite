from collections import Counter
from pathlib import Path

from .tokenizer import GraphiteTokenizer


DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


BYTE_TOKEN_START = 256
BYTE_VOCAB_SIZE = 256


def load_corpus(
    data_directory: str | Path,
) -> str:
    """Load supported text files into one training corpus."""

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_directory}"
        )

    texts = []

    supported_extensions = {
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

    for path in sorted(data_directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_extensions:
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


def byte_token(value: int) -> str:
    """Return the vocabulary representation of a byte."""

    return f"<byte:{value}>"


def build_initial_vocabulary(
    special_tokens: dict[str, int],
) -> dict[str, int]:
    """Create the mandatory special-token and byte vocabulary."""

    vocabulary = dict(special_tokens)

    used_ids = set(vocabulary.values())

    next_id = max(
        used_ids,
        default=-1,
    ) + 1

    for value in range(BYTE_VOCAB_SIZE):
        token = byte_token(value)

        if token in vocabulary:
            continue

        while next_id in used_ids:
            next_id += 1

        vocabulary[token] = next_id
        used_ids.add(next_id)

        next_id += 1

    return vocabulary


def encode_words(
    corpus: str,
) -> dict[str, list[str]]:
    """
    Convert whitespace-delimited corpus segments into byte symbols.

    Whitespace is intentionally preserved because it is meaningful
    training data. Each whitespace-delimited segment is independently
    represented as UTF-8 byte symbols.
    """

    words = corpus.split()

    return {
        word: [
            byte_token(value)
            for value in word.encode("utf-8")
        ]
        for word in words
    }


def build_word_frequency(
    corpus: str,
) -> Counter[str]:
    """Count repeated whitespace-delimited corpus segments."""

    return Counter(
        corpus.split()
    )


def count_pairs(
    word_symbols: dict[str, list[str]],
    word_frequency: Counter[str],
) -> Counter[tuple[str, str]]:
    """Count adjacent BPE symbol pairs weighted by frequency."""

    pair_counts = Counter()

    for word, symbols in word_symbols.items():
        frequency = word_frequency[word]

        for index in range(
            len(symbols) - 1
        ):
            pair = (
                symbols[index],
                symbols[index + 1],
            )

            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(
    word_symbols: dict[str, list[str]],
    pair: tuple[str, str],
) -> None:
    """Apply one BPE merge to every affected word."""

    left, right = pair
    merged_symbol = left + right

    for word, symbols in word_symbols.items():
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

        word_symbols[word] = merged


def train_bpe(
    corpus: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> GraphiteTokenizer:
    """
    Train a byte-level BPE tokenizer.

    The tokenizer always starts with:
        - special tokens
        - all 256 possible byte values

    Additional vocabulary entries are learned through BPE merges.
    """

    if vocab_size < (
        len(DEFAULT_SPECIAL_TOKENS)
        + BYTE_VOCAB_SIZE
    ):
        raise ValueError(
            "vocab_size must be at least "
            f"{len(DEFAULT_SPECIAL_TOKENS) + BYTE_VOCAB_SIZE} "
            "for byte-level BPE."
        )

    special_tokens = (
        special_tokens.copy()
        if special_tokens is not None
        else DEFAULT_SPECIAL_TOKENS.copy()
    )

    vocabulary = build_initial_vocabulary(
        special_tokens
    )

    word_frequency = build_word_frequency(
        corpus
    )

    if not word_frequency:
        raise ValueError(
            "Corpus contains no usable text."
        )

    word_symbols = encode_words(
        corpus
    )

    merges: list[tuple[str, str]] = []

    while len(vocabulary) < vocab_size:
        pair_counts = count_pairs(
            word_symbols,
            word_frequency,
        )

        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        merged_symbol = (
            best_pair[0]
            + best_pair[1]
        )

        if merged_symbol in vocabulary:
            merge_pair(
                word_symbols,
                best_pair,
            )

            continue

        vocabulary[
            merged_symbol
        ] = len(vocabulary)

        merges.append(
            best_pair
        )

        merge_pair(
            word_symbols,
            best_pair,
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
) -> GraphiteTokenizer:
    """Train and save the Graphite tokenizer."""

    corpus = load_corpus(
        data_directory
    )

    tokenizer = train_bpe(
        corpus=corpus,
        vocab_size=vocab_size,
    )

    tokenizer.save(
        output_path
    )

    return tokenizer


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

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