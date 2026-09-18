import json
from pathlib import Path
import sys


MODEL_ROOT = Path(__file__).resolve().parents[1]

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from tokenizer.tokenizer import GraphiteTokenizer


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}


def discover_files(
    data_directory: str | Path,
) -> list[Path]:
    """
    Find supported dataset files recursively.
    """

    data_directory = Path(data_directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_directory}"
        )

    return sorted(
        path
        for path in data_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )


def load_file(path: Path) -> str:
    """
    Load a supported dataset file.
    """

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def normalize_text(text: str) -> str:
    """
    Normalize raw text into a consistent representation.
    """

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    text = "\n".join(lines)

    return text.strip()


def extract_json_text(
    path: Path,
    text: str,
) -> str:
    """
    Extract readable strings from JSON or JSONL.
    """

    try:
        if path.suffix.lower() == ".jsonl":
            values = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            values = [
                json.loads(text)
            ]

    except json.JSONDecodeError:
        return text

    strings: list[str] = []

    def collect_strings(
        value: object,
    ) -> None:
        if isinstance(value, str):
            strings.append(value)

        elif isinstance(value, dict):
            for child in value.values():
                collect_strings(child)

        elif isinstance(value, list):
            for child in value:
                collect_strings(child)

    for value in values:
        collect_strings(value)

    return "\n".join(strings)


def process_file(
    path: Path,
) -> str:
    """
    Load and normalize one dataset file.
    """

    text = load_file(path)

    if path.suffix.lower() in {
        ".json",
        ".jsonl",
    }:
        text = extract_json_text(
            path,
            text,
        )

    return normalize_text(text)


def prepare_corpus(
    input_directory: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """
    Process dataset files into one training corpus.
    """

    input_directory = Path(
        input_directory
    )

    output_path = Path(
        output_path
    )

    files = discover_files(
        input_directory
    )

    if not files:
        raise ValueError(
            f"No supported dataset files found "
            f"in {input_directory}"
        )

    processed_documents: list[str] = []

    processed_files = 0
    skipped_files = 0

    for path in files:
        try:
            text = process_file(path)

        except OSError:
            skipped_files += 1
            continue

        if not text:
            skipped_files += 1
            continue

        processed_documents.append(text)
        processed_files += 1

    if not processed_documents:
        raise ValueError(
            "No usable text was produced "
            "from the dataset."
        )

    corpus = "\n\n".join(
        processed_documents
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        corpus,
        encoding="utf-8",
    )

    return {
        "files_discovered": len(files),
        "files_processed": processed_files,
        "files_skipped": skipped_files,
        "characters": len(corpus),
    }


def tokenize_corpus(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """
    Convert the prepared corpus into token IDs.
    """

    corpus_path = Path(
        corpus_path
    )

    tokenizer_path = Path(
        tokenizer_path
    )

    output_path = Path(
        output_path
    )

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Prepared corpus does not exist: "
            f"{corpus_path}"
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: "
            f"{tokenizer_path}"
        )

    tokenizer = GraphiteTokenizer.load(
        tokenizer_path
    )

    corpus = corpus_path.read_text(
        encoding="utf-8"
    )

    token_ids = tokenizer.encode(
        corpus,
        add_special_tokens=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(
            str(token_id)
            for token_id in token_ids
        ),
        encoding="utf-8",
    )

    return {
        "tokens": len(token_ids),
        "vocabulary_size": tokenizer.vocab_size,
    }


def main() -> None:
    """
    Prepare and tokenize the Graphite dataset.
    """

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    dataset_directory = (
        project_root
        / "data"
        / "datasets"
    )

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    corpus_path = (
        processed_directory
        / "corpus.txt"
    )

    tokenizer_path = (
        project_root
        / "tokenizer"
        / "files"
        / "tokenizer.json"
    )

    tokens_path = (
        processed_directory
        / "tokens.txt"
    )

    corpus_stats = prepare_corpus(
        input_directory=dataset_directory,
        output_path=corpus_path,
    )

    print(
        f"Files discovered: "
        f"{corpus_stats['files_discovered']}"
    )

    print(
        f"Files processed: "
        f"{corpus_stats['files_processed']}"
    )

    print(
        f"Files skipped: "
        f"{corpus_stats['files_skipped']}"
    )

    print(
        f"Corpus written to: "
        f"{corpus_path}"
    )

    token_stats = tokenize_corpus(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        output_path=tokens_path,
    )

    print(
        f"Tokens generated: "
        f"{token_stats['tokens']}"
    )

    print(
        f"Vocabulary size: "
        f"{token_stats['vocabulary_size']}"
    )

    print(
        f"Token IDs written to: "
        f"{tokens_path}"
    )


if __name__ == "__main__":
    main()