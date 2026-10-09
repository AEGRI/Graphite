from pathlib import Path
import json


class GraphiteTokenizer:
    """
    Byte-level BPE tokenizer for Graphite.

    Vocabulary IDs represent:
        - special tokens
        - individual UTF-8 bytes
        - learned BPE combinations of byte tokens

    Decoding uses UTF-8 replacement handling so arbitrary generated
    byte sequences cannot crash text output.
    """

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: dict[str, int] | None = None,
    ):
        self.vocabulary = vocabulary
        self.merges = merges
        self.special_tokens = special_tokens or {}

        self.id_to_token = {
            token_id: token
            for token, token_id in vocabulary.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

    @property
    def vocab_size(self) -> int:
        """Return the total tokenizer vocabulary size."""
        return len(self.vocabulary)

    def _byte_token(self, value: int) -> str:
        """Convert a byte value into its vocabulary token."""
        return f"<byte:{value}>"

    def _encode_bytes(self, text: str) -> list[str]:
        """Convert UTF-8 text into byte-level symbols."""
        return [
            self._byte_token(value)
            for value in text.encode("utf-8")
        ]

    def _apply_bpe(self, symbols: list[str]) -> list[str]:
        """Apply learned BPE merges to a symbol sequence."""
        if len(symbols) <= 1:
            return symbols

        while len(symbols) > 1:
            best_pair = None
            best_rank = None

            for index in range(len(symbols) - 1):
                pair = (
                    symbols[index],
                    symbols[index + 1],
                )

                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            left, right = best_pair
            merged_symbol = left + right

            merged = []
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

            symbols = merged

        return symbols

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode UTF-8 text into token IDs."""
        token_ids = []

        if add_special_tokens:
            bos_id = self.special_tokens.get("<bos>")

            if bos_id is not None:
                token_ids.append(bos_id)

        symbols = self._apply_bpe(
            self._encode_bytes(text)
        )

        for symbol in symbols:
            token_id = self.vocabulary.get(symbol)

            if token_id is None:
                raise ValueError(
                    f"Tokenizer vocabulary is missing token: {symbol!r}"
                )

            token_ids.append(token_id)

        if add_special_tokens:
            eos_id = self.special_tokens.get("<eos>")

            if eos_id is not None:
                token_ids.append(eos_id)

        return token_ids

    def _split_merged_symbol(self, symbol: str) -> list[str]:
        """
        Split a merged byte token into its component <byte:N> symbols.
        """
        symbols = []
        index = 0

        while index < len(symbol):
            if not symbol.startswith("<byte:", index):
                raise ValueError(
                    f"Invalid byte-level token representation: {symbol!r}"
                )

            end = symbol.find(">", index)

            if end == -1:
                raise ValueError(
                    f"Malformed byte token: {symbol!r}"
                )

            value_text = symbol[index + len("<byte:"):end]

            try:
                value = int(value_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid byte value in token: {symbol!r}"
                ) from error

            if not 0 <= value <= 255:
                raise ValueError(
                    f"Byte value out of range: {value}"
                )

            symbols.append(symbol[index:end + 1])
            index = end + 1

        return symbols

    def _decode_symbol(self, symbol: str) -> bytes:
        """Convert a vocabulary symbol into raw bytes."""
        byte_symbols = self._split_merged_symbol(symbol)
        output = bytearray()

        for byte_symbol in byte_symbols:
            start = len("<byte:")
            end = byte_symbol.find(">")

            value = int(byte_symbol[start:end])
            output.append(value)

        return bytes(output)

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs into text.

        Invalid UTF-8 is replaced with U+FFFD instead of raising an
        exception. This keeps the CLI response pipeline operational
        when the model generates arbitrary byte sequences.
        """
        special_token_ids = set(self.special_tokens.values())
        output = bytearray()

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue

            token = self.id_to_token.get(token_id)

            if token is None:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            output.extend(self._decode_symbol(token))

        return output.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save the tokenizer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 2,
            "type": "byte_bpe",
            "vocabulary": self.vocabulary,
            "merges": [
                list(pair)
                for pair in self.merges
            ],
            "special_tokens": self.special_tokens,
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphiteTokenizer":
        """Load a tokenizer from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file does not exist: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if data.get("type") != "byte_bpe":
            raise ValueError(
                f"Unsupported tokenizer type: {data.get('type')!r}"
            )

        merges = [
            tuple(pair)
            for pair in data["merges"]
        ]

        return cls(
            vocabulary=data["vocabulary"],
            merges=merges,
            special_tokens=data.get("special_tokens", {}),
        )
