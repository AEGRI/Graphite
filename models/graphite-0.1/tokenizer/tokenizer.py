from pathlib import Path
import json


class GraphiteTokenizer:
    """
    BPE tokenizer interface for Graphite 0.1.

    Loads a trained tokenizer vocabulary and merges, then converts
    text to token IDs and token IDs back into text.
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
        return len(self.vocabulary)

    def _apply_bpe(self, word: str) -> list[str]:
        """
        Apply learned BPE merges to a single word.
        """

        if not word:
            return []

        symbols = list(word)

        while len(symbols) > 1:
            pairs = [
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            ]

            available_pairs = [
                pair
                for pair in pairs
                if pair in self.merge_ranks
            ]

            if not available_pairs:
                break

            best_pair = min(
                available_pairs,
                key=lambda pair: self.merge_ranks[pair],
            )

            merged: list[str] = []
            index = 0

            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and (
                        symbols[index],
                        symbols[index + 1],
                    )
                    == best_pair
                ):
                    merged.append(
                        symbols[index]
                        + symbols[index + 1]
                    )
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
        """
        Convert text into token IDs.
        """

        token_ids: list[int] = []

        if add_special_tokens:
            bos_id = self.special_tokens.get("<bos>")

            if bos_id is not None:
                token_ids.append(bos_id)

        unknown_id = self.special_tokens.get("<unk>")

        for word in text.split():
            subwords = self._apply_bpe(word)

            for subword in subwords:
                token_id = self.vocabulary.get(subword)

                if token_id is None:
                    if unknown_id is None:
                        raise ValueError(
                            f"Unknown token with no <unk> token: "
                            f"{subword!r}"
                        )

                    token_id = unknown_id

                token_ids.append(token_id)

        if add_special_tokens:
            eos_id = self.special_tokens.get("<eos>")

            if eos_id is not None:
                token_ids.append(eos_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back into text.
        """

        special_token_ids = set(
            self.special_tokens.values()
        )

        tokens: list[str] = []

        for token_id in token_ids:
            if (
                skip_special_tokens
                and token_id in special_token_ids
            ):
                continue

            token = self.id_to_token.get(token_id)

            if token is None:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            tokens.append(token)

        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer vocabulary, merges, and special tokens.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
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
    def load(
        cls,
        path: str | Path,
    ) -> "GraphiteTokenizer":
        """
        Load a trained tokenizer.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file does not exist: {path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        merges = [
            tuple(pair)
            for pair in data["merges"]
        ]

        return cls(
            vocabulary=data["vocabulary"],
            merges=merges,
            special_tokens=data.get(
                "special_tokens",
                {},
            ),
        )