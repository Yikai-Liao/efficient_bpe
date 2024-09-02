"""
Implements a dummy class for HuggingFace BPE Tokenizer and Trainer without any pre-tokenization.
The interface is simplified and only include the key methods needed for the training and encoding/decoding.

This is implementation will be used as a prototype for the final rust implementation.
"""
from typing import List, Union, Iterable, Optional
from array import array


class BpeTokenizer:
    """
    Basic dummy class for HuggingFace BPE Tokenizer
    """

    def save(self, path: str):
        pass

    @classmethod
    def from_file(cls, path: str) -> "BpeTokenizer":
        pass

    def encode(self, sequence: str, add_special_tokens: bool = True) -> array:
        pass

    def decode(self, ids: array, skip_special_tokens: bool = True) -> str:
        pass

    def train(self, files: List[str], trainer: Optional["BpeTrainer"] = None):
        pass

    def train_from_iterator(
            self, iterator: Iterable[str],
            trainer: Optional["BpeTrainer"] = None,
            length: Optional[int] = None
    ):
        pass


class BpeTrainer:
    """
    Basic dummy class for HuggingFace BPE Trainer
    """

    def __init__(
            self, vocab_size: int,
            min_frequency: int,
            special_tokens: List[str],
            limit_alphabet: int,
            initial_alphabet: List[str],
            continuing_subword_prefix: str,
            end_of_word_suffix: str,
            show_progress: bool,
            max_piece_length: int,
    ):
        """
        :param vocab_size: The size of the final vocabulary, including all tokens and alphabet.
        :param min_frequency: The minimum frequency a pair should have in order to be merged.
        :param special_tokens: Whether to show progress bars while training.
        :param limit_alphabet:  A list of special tokens the model should know of.
        :param initial_alphabet: The maximum different characters to keep in the alphabet.
        :param continuing_subword_prefix: A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept.
        :param end_of_word_suffix: A prefix to be used for every subword that is not a beginning-of-word.
        :param show_progress: A suffix to be used for every subword that is an end-of-word.
        :param max_piece_length: Prevents creating tokens longer than the specified size. This can help with reducing polluting your vocabulary with highly repetitive tokens like ==== for wikipedia
        """
