from __future__ import annotations

import re
import json

from pathlib import Path
from typing import Dict, Iterable, List

from transformers import PreTrainedTokenizer


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int],
                 unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "</s>",
                 characters_to_replace: str = "UZOB",
                 replacement_character: str = "X",
                 uses_whitespaces: bool = False):
        # Define vocabulary mapping amino acids & special tokens
        self.vocab = vocab

        # Reverse vocabulary for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Set special tokens explicitly
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        # Set preprocessing
        self.preprocessing_strategy = lambda sequences: self._get_preprocessing_strategy(sequences,
                                                                                         characters_to_replace,
                                                                                         replacement_character,
                                                                                         uses_whitespaces)
        # Initialize parent class properly
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            vocab=vocab
        )

    @staticmethod
    def _get_preprocessing_strategy(sequences: Iterable[str],
                                    characters_to_replace,
                                    replacement_character,
                                    uses_whitespaces) -> List[str]:
        # Remove replacement characters
        sequences_cleaned = [re.sub(fr"[{characters_to_replace}]", replacement_character, sequence)
                             for sequence in sequences]

        join_char = " " if uses_whitespaces else ""
        sequences_with_spaces = [join_char.join(list(sequence)) for sequence in sequences_cleaned]
        return sequences_with_spaces

    @classmethod
    def from_config(cls, config_path: Path) -> CustomTokenizer:
        """
        Initialize a CustomTokenizer from a config file.

        Args:
            config_path (Path): Path to the JSON config file

        Returns:
            CustomTokenizer: Initialized tokenizer instance
        """

        # Read the config file
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Ensure vocab is present
        if 'vocab' not in config:
            raise ValueError("Tokenizer config file must contain 'vocab' dictionary")

        # Get required vocab
        vocab = config['vocab']
        if not isinstance(vocab, dict):
            raise ValueError("Tokenizer config file must contain 'vocab' dictionary")
        if not all(isinstance(key, str) for key in vocab.keys()):
            raise ValueError("'vocab' dictionary keys in tokenizer config must all be strings")
        if not all(isinstance(value, int) for value in vocab.values()):
            raise ValueError("'vocab' dictionary values in tokenizer config must all be integers")

        # Get optional parameters with defaults
        unk_token = config.get('unk_token', "<unk>")
        pad_token = config.get('pad_token', "<pad>")
        eos_token = config.get('eos_token', "</s>")
        characters_to_replace = config.get('characters_to_replace', "UZOB")
        replacement_character = config.get('replacement_character', "X")
        uses_whitespaces = config.get('uses_whitespaces', False)

        # Initialize and return the tokenizer
        return cls(
            vocab=vocab,
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            characters_to_replace=characters_to_replace,
            replacement_character=replacement_character,
            uses_whitespaces=uses_whitespaces
        )

    def get_vocab(self):
        """ Returns the vocabulary dictionary. """
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        # Reverse the vocab to convert id back to token
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return reverse_vocab.get(index, self.unk_token)
