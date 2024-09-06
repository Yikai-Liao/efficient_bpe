"""
Implements a dummy class for HuggingFace BPE Tokenizer and Trainer without any pre-tokenization.
The interface is simplified and only include the key methods needed for the training and encoding/decoding.

This is implementation will be used as a prototype for the final rust implementation.
"""
from tqdm import tqdm
from itertools import chain
from collections import defaultdict, Counter
from typing import List, Dict, Union, Iterable, Optional, Tuple

import numpy as np
from numpy import uint32 as u32


class BpeTokenizer:
    """
    Basic dummy class for HuggingFace BPE Tokenizer
    """

    def save(self, path: str):
        pass

    @classmethod
    def from_file(cls, path: str) -> "BpeTokenizer":
        pass

    def encode(self, sequence: str, add_special_tokens: bool = True) -> np.ndarray[u32]:
        pass

    def decode(self, ids: np.ndarray[u32], skip_special_tokens: bool = True) -> str:
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
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = initial_alphabet
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.show_progress = show_progress
        assert show_progress is True, "Only show_progress=True is supported"
        self.max_piece_length = max_piece_length

    def do_train(
            self, word_counts: Dict[str, int]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        :return: The final merges and the final vocabulary
        """
        id2word, word2id = self.compute_alphabet(word_counts)
        print('id2word:', id2word)
        print('word2id:', word2id)
        corpus, seg_status, freq_status, pair_freq, pair_pos = self.init_bpe_satus(word_counts, word2id)

        return [('@@', '@@')], id2word

    def compute_alphabet(self, word_counts: Dict[str, int]) -> Tuple[List[str], Dict[str, int]]:
        """
        Compute the alphabet from the word counts
        :param word_counts: The word counts
        :return: The id2word and word2id mappings
        """
        alphabet = defaultdict(int)
        for word, freq in word_counts.items():
            for c in word:
                alphabet[c] += freq
        for c in self.initial_alphabet:
            alphabet[c] += 1 << 30

        chars = sorted(alphabet.keys(), key=lambda x: alphabet[x], reverse=True)
        if self.limit_alphabet:
            chars = chars[:self.limit_alphabet]

        word2id = {word: i for i, word in enumerate(chain(self.special_tokens, chars))}
        return self.special_tokens + chars, word2id

    @staticmethod
    def init_bpe_satus(
            word_counts: Dict[str, int],
            word2id: Dict[str, int]
    ):

        words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        print(words)
        total_len = freq_len = 1
        for word, freq in words:
            cur_len = len(word) + 1
            total_len += cur_len
            if freq > 1:
                freq_len += cur_len

        corpus: np.ndarray[u32] = np.zeros(total_len, dtype=u32)
        seg_status: np.ndarray[u32] = np.ones(total_len, dtype=u32)
        freq_status: np.ndarray[u32] = np.zeros(freq_len, dtype=u32)

        seg_status[0] = 0
        prev = 1
        prev_token = -1
        pair_freq = defaultdict(int)
        pair_pos = defaultdict(list)
        pbar = tqdm(total = total_len, desc='Initializing BPE status', leave=True)
        pbar.update()

        for word, freq in words:
            # All words are separated by a '0' token, with '0' at both beginning and end of corpus
            # 0, char_1, char_2, ..., char_n, 0
            # word_end points to the 0 after the word
            word_end = prev + len(word)
            # set the seg status at the margin of the word to 0
            seg_status[word_end] = 0
            # set the freq status for the word if freq > 1
            if freq > 1:
                freq_status[prev:word_end] = freq
            # build the corpus with the word ids & update the pair_freq
            for i, c in enumerate(word, start=prev):
                c_id = word2id.get(c, -1)
                if c_id >= 0:
                    # known token
                    corpus[i] = c_id
                    if prev_token >= 0:
                        # update the pair_freq and pair_pos
                        pair_freq[(prev_token, c_id)] += freq
                        pair_pos[(prev_token, c_id)].append(i - 1)
                else:
                    # unknown token, just ignore
                    corpus[i] = seg_status[i] = 0
                prev_token = c_id
                pbar.update()
            # prev points to the beginning of the next word
            prev = word_end + 1
            prev_token = -1
            pbar.update()
        pbar.close()

        print('corpus:', corpus)
        print('seg_status:', seg_status)
        print('freq_status:', freq_status)
        print('pair_freq:', pair_freq)
        print('pair_pos:', pair_pos)
        return corpus, seg_status, freq_status, pair_freq, pair_pos

if __name__ == "__main__":
    trainer = BpeTrainer(
        vocab_size=1000,
        min_frequency=2,
        special_tokens=['<PAD>', '<UNK>'],
        limit_alphabet=100,
        initial_alphabet=list('abcdefghijklmnopqrstuvwxyz'),
        continuing_subword_prefix='@@',
        end_of_word_suffix='@@',
        show_progress=True,
        max_piece_length=16
    )
    tokenizer = BpeTokenizer()
    word_counts = Counter('the quick brown fox jumps over the lazy dog oh dog'.split(' '))

    merges, vocab = trainer.do_train(word_counts)

