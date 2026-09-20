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


def discover_files(
    data_directory: str | Path,
) -> list[Path]:
    """Find all supported dataset files."""

    data_directory = Path(
        data_directory
    )

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_directory}"
        )

    return sorted(
        path
        for path in data_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def load_file(
    path: Path,
) -> str:
    """Read a dataset file as UTF-8."""

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def normalize_text(
    text: str,
) -> str:
    """
    Normalize line endings without destroying meaningful
    whitespace or indentation.
    """

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    return text.strip()


def extract_json_text(
    path: Path,
    text: str,
) -> str:
    """
    Extract string values from JSON or JSONL files.

    This allows structured metadata files to contribute their
    textual content without training directly on JSON syntax.
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

    strings = []

    def collect_strings(
        value: object,
    ) -> None:
        if isinstance(
            value,
            str,
        ):
            strings.append(value)

        elif isinstance(
            value,
            dict,
        ):
            for child in value.values():
                collect_strings(child)

        elif isinstance(
            value,
            list,
        ):
            for child in value:
                collect_strings(child)

    for value in values:
        collect_strings(value)

    return "\n".join(
        strings
    )


def process_file(
    path: Path,
) -> str:
    """Load and normalize one dataset file."""

    text = load_file(
        path
    )

    if path.suffix.lower() in {
        ".json",
        ".jsonl",
    }:
        text = extract_json_text(
            path,
            text,
        )

    return normalize_text(
        text
    )


def prepare_documents(
    input_directory: str | Path,
) -> tuple[list[str], dict[str, int]]:
    """
    Process every source document independently.

    Keeping documents separate allows us to insert EOS boundaries
    between documents during tokenization.
    """

    input_directory = Path(
        input_directory
    )

    files = discover_files(
        input_directory
    )

    if not files:
        raise ValueError(
            f"No supported files found in: "
            f"{input_directory}"
        )

    documents = []

    processed_files = 0
    skipped_files = 0

    for path in files:
        try:
            text = process_file(
                path
            )

        except OSError:
            skipped_files += 1
            continue

        if not text:
            skipped_files += 1
            continue

        documents.append(
            text
        )

        processed_files += 1

    if not documents:
        raise ValueError(
            "No usable documents were produced."
        )

    total_characters = sum(
        len(document)
        for document in documents
    )

    stats = {
        "files_discovered": len(files),
        "files_processed": processed_files,
        "files_skipped": skipped_files,
        "documents": len(documents),
        "characters": total_characters,
    }

    return documents, stats


def write_corpus(
    documents: list[str],
    output_path: str | Path,
) -> None:
    """
    Write the processed corpus while keeping document boundaries.

    Two newlines separate documents. Individual document contents
    retain their internal whitespace.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus = "\n\n".join(
        documents
    )

    output_path.write_text(
        corpus,
        encoding="utf-8",
    )


def tokenize_documents(
    documents: list[str],
    tokenizer: GraphiteTokenizer,
) -> list[int]:
    """
    Tokenize documents independently and place EOS boundaries
    between them.

    BOS/EOS are not automatically added by the tokenizer here
    because document boundaries need explicit control.
    """

    token_ids = []

    bos_id = tokenizer.special_tokens.get(
        "<bos>"
    )

    eos_id = tokenizer.special_tokens.get(
        "<eos>"
    )

    for document in documents:
        if bos_id is not None:
            token_ids.append(
                bos_id
            )

        token_ids.extend(
            tokenizer.encode(
                document,
                add_special_tokens=False,
            )
        )

        if eos_id is not None:
            token_ids.append(
                eos_id
            )

    return token_ids


def write_tokens(
    token_ids: list[int],
    output_path: str | Path,
) -> None:
    """Write token IDs as one integer per line."""

    output_path = Path(
        output_path
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


def prepare_corpus(
    input_directory: str | Path,
    corpus_output_path: str | Path,
    tokens_output_path: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, int]:
    """
    Run the complete preprocessing pipeline.

    Returns statistics for both document processing and tokenization.
    """

    documents, stats = prepare_documents(
        input_directory
    )

    write_corpus(
        documents,
        corpus_output_path,
    )

    tokenizer = GraphiteTokenizer.load(
        tokenizer_path
    )

    token_ids = tokenize_documents(
        documents,
        tokenizer,
    )

    if not token_ids:
        raise ValueError(
            "Tokenizer produced no tokens."
        )

    write_tokens(
        token_ids,
        tokens_output_path,
    )

    stats.update(
        {
            "tokens": len(token_ids),
            "vocabulary_size": tokenizer.vocab_size,
        }
    )

    return stats


def main() -> None:
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

    tokens_path = (
        processed_directory
        / "tokens.txt"
    )

    tokenizer_path = (
        project_root
        / "tokenizer"
        / "files"
        / "tokenizer.json"
    )

    stats = prepare_corpus(
        input_directory=dataset_directory,
        corpus_output_path=corpus_path,
        tokens_output_path=tokens_path,
        tokenizer_path=tokenizer_path,
    )

    print(
        f"Files discovered: "
        f"{stats['files_discovered']}"
    )

    print(
        f"Files processed: "
        f"{stats['files_processed']}"
    )

    print(
        f"Files skipped: "
        f"{stats['files_skipped']}"
    )

    print(
        f"Documents: "
        f"{stats['documents']}"
    )

    print(
        f"Characters: "
        f"{stats['characters']:,}"
    )

    print(
        f"Tokens: "
        f"{stats['tokens']:,}"
    )

    print(
        f"Vocabulary size: "
        f"{stats['vocabulary_size']:,}"
    )

    print(
        f"Corpus written to: "
        f"{corpus_path}"
    )

    print(
        f"Tokens written to: "
        f"{tokens_path}"
    )


if __name__ == "__main__":
    main()