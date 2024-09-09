"""
Implements a dummy class for HuggingFace BPE Tokenizer and Trainer without any pre-tokenization.
The interface is simplified and only include the key methods needed for the training and encoding/decoding.

This is implementation will be used as a prototype for the final rust implementation.
"""
from tqdm import tqdm
from array import array
from itertools import pairwise
from collections import defaultdict, Counter
from typing import List, Dict, Iterable, Optional, Tuple

import heapq
import bisect


class BpeTokenizer:
    """
    Basic dummy class for HuggingFace BPE Tokenizer
    """

    def save(self, path: str):
        pass

    @classmethod
    def from_file(cls, path: str) -> "BpeTokenizer":
        pass

    def encode(self, sequence: str, add_special_tokens: bool = True):
        pass

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
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
        tag_begin = self.continuing_subword_prefix is not None and self.continuing_subword_prefix != ''
        tag_end = self.end_of_word_suffix is not None and self.end_of_word_suffix != ''

        # 1. Compute the alphabet
        char2id, id2char = compute_alphabet(
            word_counts, self.initial_alphabet, self.limit_alphabet, self.special_tokens)
        token_len = [1] * len(id2char)
        # 2. Build the corpus
        corpus, freq_pivot, freq_info = build_corpus(word_counts, char2id, tag_begin, tag_end)
        # If using prefix and suffix, the true vocab is not from id2char
        seen_vocab = set(corpus) if tag_begin or tag_end else set(id2char)

        # 3. Count token pairs and get the priority queue
        pair_pos, pair_freq, queue = count_token_pairs(corpus, freq_pivot, freq_info, self.min_frequency)

        # 4. Merge tokens (Main loop)
        merges = []
        # using vocal_size + 2 because of <PAD> and <UNK>
        pbar = tqdm(total=self.vocab_size + 2 - len(seen_vocab), desc='Merging tokens')
        while len(seen_vocab) < self.vocab_size + 2:
            # 4.1 Find the most frequent combination of two tokens in the corpus
            pair, freq = most_frequent_combination(queue, pair_freq, pair_pos, self.min_frequency)
            # early stop if no pair is found
            if freq < self.min_frequency or pair is None:
                break
            # 4.2 Assign a new token (u32) to the pair
            new_token = assign_token(pair, char2id, id2char, seen_vocab, token_len)
            print(f'freq: {freq}')
            merges.append((pair, new_token))
            # 4.3 Merge the pair in the corpus
            pos_list = pair_pos.pop(pair, None)
            assert pos_list is not None, f"pair {pair} not found in pair_pos, something is wrong"
            pair_freq_patch, pair_pos_patch = \
                merge_token_pair(corpus, pair, new_token, pos_list, freq_pivot, freq_info, token_len)
            # 4.4 Update the pair_freq and pair_pos
            apply_patch(queue, pair_freq, pair_pos, pair_freq_patch, pair_pos_patch, self.min_frequency)

            show_corpus(corpus, id2char, self.continuing_subword_prefix, self.end_of_word_suffix, token_len)
            # update the progress bar
            pbar.update(1)

        pbar.close()
        print(char2id)


def compute_alphabet(
        words: Dict[str, int],
        initial_alphabet: List[str],
        limit_alphabet: int,
        special_tokens: List[str],

):
    alphabet = {'<PAD>': 2 << 59, '<UNK>': 2 << 58}

    for c in initial_alphabet:
        alphabet[c] = 1 << 50

    for c in special_tokens:
        alphabet[c] = 1 << 51

    for word, freq in words.items():
        for c in word:
            alphabet[c] = alphabet.get(c, 0) + freq

    chars = [(-f, c) for c, f in alphabet.items()]
    chars.sort()
    if limit_alphabet > 0:
        chars = chars[:limit_alphabet + 2]
    id2word = [c for _, c in chars]
    word2id = {c: i for i, c in enumerate(id2word)}
    return word2id, id2word


def build_corpus(
        words: Dict[str, int],
        word2id: Dict[str, int],
        tag_begin: bool,
        tag_end: bool
):
    assert word2id['<PAD>'] == 0 and word2id['<UNK>'] == 1

    byte_num = sum(len(word) for word in words)
    corpus_size = 1 + len(words) + byte_num
    # corpus is u32
    corpus = array('I', [0] * corpus_size)

    # sort words by frequency
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    freq_pivot = [0]
    freq_info = [words[0][1]]
    pivot = 1
    for word, freq in words:
        assert 0 < freq <= freq_info[-1]
        # update freq pivot and freq info
        if freq < freq_info[-1]:
            freq_pivot.append(pivot)
            freq_info.append(freq)

        begin = pivot
        for c in word:
            # set to <UNK> if not in word2id
            corpus[pivot] = word2id.get(c, 1)
            pivot += 1
        if tag_begin:
            # using bit operation, set the highest bit of the first character to 1
            corpus[begin] |= 1 << 31
        if tag_end:
            # using bit operation, set the second-highest bit of the last character to 1
            corpus[pivot - 1] |= 1 << 30
        pivot += 1

    freq_pivot.append(pivot + 1)
    freq_info.append(0)
    # freq_pivot and freq_info are u64
    return corpus, array('Q', freq_pivot), array('Q', freq_info)


def count_token_pairs(corpus: array, freq_pivot: array, freq_info: array, min_freq: int):
    pair_pos = defaultdict(list)
    pair_freq = defaultdict(int)

    next_pivot = freq_pivot[1]
    freq, freq_i = freq_info[0], 0

    for i, (x, y) in enumerate(pairwise(tqdm(corpus, desc='Counting token pairs'))):
        if i >= next_pivot:
            freq_i += 1
            freq = freq_info[freq_i]
            next_pivot = freq_pivot[freq_i + 1]

        if x < 2 or y < 2:
            continue
        pair = (x, y)
        pair_pos[pair].append(i)
        pair_freq[pair] += freq

    pair_freq = {p: f for p, f in pair_freq.items() if f >= min_freq}
    pair_pos = {pair: array('Q', pair_pos[pair]) for pair in pair_pos.keys()}
    # build a priority queue according to the pair_freq using heapq
    # using -f to make the priority queue a max heap
    queue = [(-f, p) for p, f in pair_freq.items()]
    heapq.heapify(queue)

    return pair_pos, pair_freq, queue


def most_frequent_combination(queue, pair_freq, pair_pos, min_freq):
    """
    Find the most frequent combination of two tokens in the corpus
    """
    while len(queue) > 0:
        cached_freq, pair = heapq.heappop(queue)
        cached_freq = -cached_freq
        ground_freq = pair_freq[pair]
        if cached_freq == ground_freq:
            return pair, ground_freq
        elif ground_freq >= min_freq:
            # push the pair back to the queue if the frequency is larger than min_freq
            heapq.heappush(queue, (-ground_freq, pair))
            # TODO: add compression for pair_pos in the future
        else:
            # remove the pair from the pair_pos and pair_freq
            pair_pos.pop(pair, None)
            pair_freq.pop(pair, None)
    return None, 0


def build_char(x: int, id2word: List[str], prefix: str, suffix: str):
    raw_char = id2word[x & 0x7FFFFFFF]
    if not x & 0x80000000:
        # prefix is for continuing subword
        raw_char = prefix + raw_char
    # suffix is for end of word
    if x & 0x40000000:
        raw_char += suffix
    return raw_char


def assign_token(pair, word2id, id2word, seen_vocab, token_len):
    """
    Assign a new token (u32) to the pair
    """
    x_str, y_str = id2word[pair[0] & 0x7FFFFFFF], id2word[pair[1] & 0x7FFFFFFF]
    pair_str = x_str + y_str
    print(x_str + '|' + y_str, end=' ')
    if pair_str in word2id:
        raw_id = word2id[pair_str]
    else:
        raw_id = len(id2word)
        id2word.append(pair_str)
        word2id[pair_str] = raw_id
        token_len.append(
            token_len[pair[0] & 0x7FFFFFFF] + token_len[pair[1] & 0x7FFFFFFF])  # update raw_id with highest two bits
    true_id = raw_id | (pair[0] & 0xC0000000) | (pair[1] & 0xC0000000)
    assert raw_id not in seen_vocab
    seen_vocab.add(raw_id)
    return true_id


def merge_token_pair(corpus, pair, new_token, pos_list, freq_pivot, freq_info, token_len):
    """
    Merge the pair in the corpus
    """
    # pos_list = pair_pos.pop(pair, None)
    # assert pos_list is not None, f"pair {pair} not found in pair_pos, something is wrong"
    pair_freq_patch = defaultdict(int)
    pair_pos_patch = defaultdict(list)

    x, y = pair
    len_x, len_y = token_len[x & 0x7FFFFFFF], token_len[y & 0x7FFFFFFF]
    len_pair = len_x + len_y

    # init frequency information
    freq_i = bisect.bisect_right(freq_pivot, pos_list[0]) - 1
    freq = freq_info[freq_i]
    next_pivot = freq_pivot[freq_i + 1]

    # main loop
    for pos in pos_list:
        pos_x, pos_y = pos, pos + len_x
        pos_end = pos_y + len_y
        # filter out invalid positions
        if corpus[pos_x] != x or corpus[pos_y] != y:
            continue
        # check freq info
        if pos >= next_pivot:
            freq_i = bisect.bisect_right(freq_pivot[freq_i + 1:], pos) + freq_i + 1
            # TODO: remove the assert after debug
            assert freq_i == bisect.bisect_right(freq_pivot, pos) - 1
            freq = freq_info[freq_i]
            next_pivot = freq_pivot[freq_i + 1]

        # merge x and y
        corpus[pos_x] = corpus[pos_end - 1] = new_token
        for i in range(pos_x + 1, pos_end - 1):
            corpus[i] = 0

        l, r = corpus[pos_x - 1], corpus[pos_y + len_y]
        # modify left
        if l > 1:  # not <PAD> or <UNK>
            pair_freq_patch[(l, x)] -= freq
            pair_freq_patch[(l, new_token)] += freq
            len_l = token_len[l & 0x7FFFFFFF]
            pair_pos_patch[(l, new_token)].append(pos_x - len_l)

        # modify right
        if r > 1:
            pair_freq_patch[(y, r)] -= freq
            pair_freq_patch[(new_token, r)] += freq
            pair_pos_patch[(new_token, r)].append(pos_x)
    return pair_freq_patch, pair_pos_patch


def apply_patch(queue, pair_freq, pair_pos, pair_freq_patch, pair_pos_patch, min_freq):
    for pair, freq in pair_freq_patch.items():
        pair_freq[pair] = pair_freq.get(pair, 0) + freq
        if freq < min_freq:
            continue
        pair_pos[pair] = array('Q', pair_pos_patch[pair])
        heapq.heappush(queue, (-freq, pair))


def show_corpus(corpus, id2char, continuing_subword_prefix, end_of_word_suffix, token_len):
    i = 0
    final_corpus = []
    while i < len(corpus):
        final_corpus.append(build_char(corpus[i], id2char, continuing_subword_prefix, end_of_word_suffix))
        i += token_len[corpus[i] & 0x7FFFFFFF]
    print(" ".join(final_corpus))


if __name__ == "__main__":
    trainer = BpeTrainer(
        vocab_size=1000,
        min_frequency=1,
        special_tokens=['<PAD>', '<UNK>'],
        limit_alphabet=100,
        initial_alphabet=[],
        continuing_subword_prefix='',
        end_of_word_suffix='',
        show_progress=True,
        max_piece_length=16
    )
    tokenizer = BpeTokenizer()
    word_counts = Counter(
        'the quick brown fox jumps over the lazy dog oh dog 00000001111111000001010101000111'.split(' '))

    trainer.do_train(word_counts)
