from collections import Counter
from pathlib import Path

from .tokenizer import GraphiteTokenizer


DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


def load_corpus(data_directory: str | Path) -> str:
    """
    Load text files from a directory and combine them into one corpus.
    """

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_directory}"
        )

    texts: list[str] = []

    for path in sorted(data_directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".txt",
            ".md",
            ".json",
            ".jsonl",
        }:
            continue

        texts.append(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    if not texts:
        raise ValueError(
            f"No supported text files found in {data_directory}"
        )

    return "\n".join(texts)


def build_word_frequency(
    corpus: str,
) -> Counter[str]:
    """
    Count word frequencies in the corpus.
    """

    return Counter(corpus.split())


def build_initial_vocabulary(
    word_frequency: Counter[str],
    special_tokens: dict[str, int],
) -> dict[str, int]:
    """
    Create the initial character-level BPE vocabulary.
    """

    vocabulary = dict(special_tokens)

    next_id = (
        max(vocabulary.values(), default=-1) + 1
    )

    characters = Counter()

    for word, frequency in word_frequency.items():
        for character in word:
            characters[character] += frequency

    for character in sorted(
        characters,
        key=lambda value: (-characters[value], value),
    ):
        vocabulary[character] = next_id
        next_id += 1

    return vocabulary


def build_word_symbols(
    word_frequency: Counter[str],
) -> dict[str, list[str]]:
    """
    Represent each word as a sequence of characters.
    """

    return {
        word: list(word)
        for word in word_frequency
    }


def count_pairs(
    word_symbols: dict[str, list[str]],
    word_frequency: Counter[str],
) -> Counter[tuple[str, str]]:
    """
    Count adjacent symbol pairs weighted by word frequency.
    """

    pair_counts: Counter[tuple[str, str]] = Counter()

    for word, symbols in word_symbols.items():
        frequency = word_frequency[word]

        for index in range(len(symbols) - 1):
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
    """
    Merge one BPE pair throughout the vocabulary.
    """

    left, right = pair
    merged_symbol = left + right

    for word, symbols in word_symbols.items():
        merged: list[str] = []
        index = 0

        while index < len(symbols):
            if (
                index < len(symbols) - 1
                and symbols[index] == left
                and symbols[index + 1] == right
            ):
                merged.append(merged_symbol)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        word_symbols[word] = merged


def train_bpe(
    corpus: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> GraphiteTokenizer:
    """
    Train a basic Byte Pair Encoding tokenizer.
    """

    special_tokens = (
        special_tokens
        or DEFAULT_SPECIAL_TOKENS.copy()
    )

    word_frequency = build_word_frequency(
        corpus
    )

    if not word_frequency:
        raise ValueError(
            "Corpus contains no usable words."
        )

    vocabulary = build_initial_vocabulary(
        word_frequency,
        special_tokens,
    )

    word_symbols = build_word_symbols(
        word_frequency
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
            best_pair[0] + best_pair[1]
        )

        if merged_symbol in vocabulary:
            merge_pair(
                word_symbols,
                best_pair,
            )
            continue

        vocabulary[merged_symbol] = len(
            vocabulary
        )

        merges.append(best_pair)

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
    """
    Train and save the Graphite tokenizer.
    """

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
    """
    Command-line entry point.
    """

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
    )

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Tokenizer written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()