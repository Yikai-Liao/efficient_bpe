import sys
from array import array
from collections import defaultdict, OrderedDict
from itertools import chain
from copy import deepcopy
import json
import heapq
from tqdm import tqdm
from typing import Optional, Set, Union, List, Iterable
import re

try:
    from itertools import pairwise
    print("Using itertools.pairwise")
except:
    print("Using custom pairwise")
    from itertools import tee
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def count_unit(data):
    count = 0
    for d in data:
        if len(d) == 1:
            count += 1
    return count


def preproc_idx(full_indices, word_a, word_b):
    # 获取indices
    if word_a == word_b:  # 特殊情况处理
        # 例如 对于“1 0 0 0”这种情况合并 0和0, 从后向前后并，得到“1 0 00”
        len_a = len(word_a)
        indices = full_indices[::-1]
        new_indices = [indices[0]]
        for idx in indices[1:]:
            if new_indices[-1] - idx != len_a:
                new_indices.append(idx)
        indices = array('I', new_indices[::-1])
        return indices
    else:
        return full_indices

class BPE:
    def __init__(self, vocab: Optional[Set[str]]) -> None:
        self.vocab = vocab if vocab else {}
        self.max_len = max(map(len, self.vocab)) if self.vocab else 0

    def decode_forward(self, data: str):
        i, j = 0, min(self.max_len, len(data))
        res = []
        while i < len(data):
            while j > i:
                if data[i:j] in self.vocab:
                    res.append(data[i:j])
                    i = j
                    j = min(i + self.max_len, len(data))
                    break
                else:
                    j -= 1
            else:
                res.append(data[i])
                i += 1
                j = min(i + self.max_len, len(data))
        return res

    def decode_backward(self, data: str):
        i, j = len(data), max(len(data) - self.max_len, 0)
        res = []
        while i > 0:
            while j < i:
                if data[j:i] in self.vocab:
                    res.append(data[j:i])
                    i = j
                    j = max(i - self.max_len, 0)
                    break
                else:
                    j += 1
            else:
                res.append(data[i - 1])
                i -= 1
                j = max(i - self.max_len, 0)
        res.reverse()
        return res

    def decode_bidirectional(self, data: str):
        res = []
        for d in data.split("\n"):
            forward = self.decode_forward(d)
            backward = self.decode_backward(d)
            if len(forward) < len(backward):
                res.extend(forward)
            elif len(forward) > len(backward):
                res.extend(backward)
            elif count_unit(forward) < count_unit(backward):
                res.extend(forward)
            else:
                res.extend(backward)
            res.append("\n")
        return res[:-1]

    def tokenize(self, text: str) -> List[str]:
        text = f"#{text}#"
        word_pair_pos, pair_freq_queue, seg_status = self.init_count(text)
        while len(pair_freq_queue) > 0:
            pair = self.most_frequent_combination(pair_freq_queue)
            self.merge(pair, word_pair_pos, seg_status, pair_freq_queue, text)

        i, ans = 1, []
        while i < len(text) - 1:
            j = i + seg_status[i]
            ans.append(text[i: j])
            i = j
        return ans

    def init_count(self, text: str):
        word_pair_pos = defaultdict(set)
        for i, (pre_char, nxt_char) in enumerate(pairwise(text), start=0):
            word_pair_pos[(pre_char, nxt_char)].add(i)
        
        word_pair_pos_valid, pair_freq_queue = defaultdict(set), []
        for word_pair, indices in word_pair_pos.items():
            if "".join(word_pair) not in self.vocab:
                continue
            word_pair_pos_valid[word_pair] = indices
            pair_freq_queue.append((-self.vocab["".join(word_pair)], word_pair))
        word_pair_pos = word_pair_pos_valid
        heapq.heapify(pair_freq_queue)
        seg_status = array('B', [1] * len(text))
        return word_pair_pos, pair_freq_queue, seg_status
    
    @staticmethod
    def most_frequent_combination(queue):
        return heapq.heappop(queue)[1]

    def merge(self, pair, word_pair_pos, seg_status, queue,text):
        (word_a, word_b), word_comb = pair, "".join(pair)
        len_a, len_b, len_comb = len(word_a), len(word_b), len(word_comb)
        indices = word_pair_pos.pop(pair)
        if word_a == word_b:
            indices = sorted(indices, reverse=True)
            new_indices = [indices[0]]
            for idx in indices[1:]:
                if new_indices[-1] - idx != len_a:
                    new_indices.append(idx)
            indices = new_indices

        new_pairs = defaultdict(set)
        for idx in indices:
            if not (seg_status[idx] == len_a and seg_status[idx + len_a] == len_b):
                continue
            seg_status[idx] = len_comb
            seg_status[idx + len_a] = 0
            seg_status[idx + len_comb - 1] = len_comb

            pre_end, nxt_start = idx, idx + len_comb
            nxt_end = seg_status[nxt_start] + nxt_start
            pre_start = idx - seg_status[idx - 1]
            pre_word, nxt_word = text[pre_start: pre_end], text[nxt_start: nxt_end]
            pre_pair, nxt_pair = (pre_word, word_comb), (word_comb, nxt_word)

            if "".join(pre_pair) in self.vocab:
                new_pairs[pre_pair].add(pre_start)
            if "".join(nxt_pair) in self.vocab:
                new_pairs[nxt_pair].add(pre_end)

        for pair, indices in new_pairs.items():
            word_pair_pos[pair] = indices
            heapq.heappush(queue, (-self.vocab["".join(pair)], pair))

    def dump_vocab(self, path):
        print(f"Dump Vocab as json file to {path}")
        with open(path, "w") as f:
            json.dump(OrderedDict((
                sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
            )), f, ensure_ascii=False, indent=4)
    
    def load_vocab(self, path):
        with open(path) as f:
            self.vocab = json.load(f)

class BPETrainer:
    def __init__(self,vocab_size, min_freq: int = 10, compress_threshold: float = 0, single_char: bool=True) -> None:
        self.corpus = ""
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.compress_threshold = compress_threshold
        self.corpus = self.word_pair_pos = self.word_pair_len = None
        self.seg_status= self.vocab = self.pair_freq_queue = self.word_count = None
        self.single_char = single_char

    def train_from_file(self, path: Union[List[str], str], verbose: bool = False) -> BPE:
        return self.train_from_iter(self.load_file(path, verbose), verbose)
    
    def train_from_iter(self, data: Iterable[str], verbose:bool = False) -> BPE:
        self.init_word_pair(data, self.min_freq, verbose)
        self.epoch = 0
        while len(self.vocab) < self.vocab_size:
            comb, freq = self.most_frequent_combination()
            if freq <= self.min_freq:
                break
            self.merge_word(comb, freq)
            if verbose and (freq > int(1e5) or self.epoch % 50 == 0):
                self.log(self.epoch, comb, freq)
            word_comb = "".join(comb)
            self.vocab[word_comb] = self.word_count[word_comb] = freq
            self.epoch += 1
        if verbose:
            print(f"Final Vocab ({'' if self.single_char else 'Not'} Including Single Characters) Size: {len(self.vocab)} in {self.epoch} Epoch")
        return BPE(self.vocab)

    @classmethod
    def load_file(cls, path: Union[List[str], str], verbose: bool):
        path = path if isinstance(path ,list) else [path]
        pbar = tqdm(path) if verbose else path
        for p in pbar:
            with open(p, "r") as f:
                yield f.read()
    
    def init_word_pair(self, corpus_iter, min_freq: int, verbose: bool):
        # 简单的统计字符的组合频率
        corpus = self.replace_punc("\n".join(chain([''], corpus_iter, [''])), verbose)
        assert 4294967295 > len(corpus)
        word_pair_pos = defaultdict(lambda: array('I'))
        vocab = defaultdict(int)
        if verbose:
            print("Initing pair count...")
        pbar = enumerate(pairwise(corpus), start=0)

        pbar = pbar if not verbose else tqdm(pbar, total=len(corpus)-1)
        if self.single_char:
            for i, (pre_char, nxt_char) in pbar:
                if pre_char == "#":
                    continue
                vocab[pre_char] += 1
                if nxt_char == "#":
                    continue
                word_pair_pos[(pre_char, nxt_char)].append(i)
        else:
            for i, (pre_char, nxt_char) in pbar:
                if pre_char == "#" or nxt_char == "#":
                    continue
                word_pair_pos[(pre_char, nxt_char)].append(i)
        
        self.word_pair_pos = {
                word_pair: indices
                for word_pair, indices in word_pair_pos.items()
                if len(indices) >= min_freq
            }
        self.word_pair_len = {
            word_pair: len(indices)
            for word_pair, indices in self.word_pair_pos.items()
        }
        self.pair_freq_queue = [(-freq, word_pair) for word_pair, freq in self.word_pair_len.items()]
        heapq.heapify(self.pair_freq_queue)
        self.vocab = vocab
        self.corpus = corpus
        self.word_count = deepcopy(self.vocab)
        self.seg_status = array('B', [1] * (len(self.corpus) + 2))
        if verbose:
            print("init finish!")

    def log(self, epoch, comb, freq):
        print(f"epoch: {epoch}\tcomb: {' + '.join(comb)}\tfreq: {freq}")

    
    def most_frequent_combination(self):
        # 找到频率最高的组合
        while len(self.pair_freq_queue) > 0:
            cached_freq, comb = heapq.heappop(self.pair_freq_queue)
            ground_freq = self.word_pair_len[comb]
            cached_freq = -cached_freq
            if cached_freq == ground_freq:
                return comb, cached_freq
            elif ground_freq > self.min_freq:
                heapq.heappush(self.pair_freq_queue, (-ground_freq, comb))
                if ground_freq / len(self.word_pair_pos[comb]) < self.compress_threshold:
                    self.word_pair_pos[comb] = self.compress_indices(*comb) 
            else:
                self.word_pair_len.pop(comb, None)
                self.word_pair_pos.pop(comb, None)
        return None, 0
    
    def compress_indices(self, word_a, word_b, indices=None):
        # 压缩索引
        seg = self.seg_status
        len_a, len_b = len(word_a), len(word_b)
        indices = indices or self.word_pair_pos[(word_a, word_b)]
        return array('I', (
            i for i in indices
            if seg[i] == len_a and seg[i + len_a] == len_b
        ))


    def merge_word(self, comb, freq):
        # 加速访问
        word_a, word_b = comb  # 这里因为还要查找comb前后的词，所以不能直接使用pre_word, nxt_word
        word_comb = word_a + word_b
        len_a, len_b, len_comb = len(word_a), len(word_b), len(word_comb)
        seg_status = self.seg_status
        word_pair_v2 = self.word_pair_pos
        corpus = self.corpus
        indices = word_pair_v2[comb]
        if len(indices) > freq:
            indices = self.compress_indices(word_a, word_b, indices) 
        indices = preproc_idx(indices, word_a, word_b)

        self.word_count[word_a] -= len(indices)
        self.word_count[word_b] -= len(indices)
        new_pairs = defaultdict(list)
        for i in indices:
            pre_end, nxt_start = i, i + len_comb
            nxt_end = seg_status[nxt_start] + nxt_start
            pre_start = i - seg_status[i - 1]

            pre_word, nxt_word = corpus[pre_start: pre_end], corpus[nxt_start: nxt_end]
            
            if pre_word != "#":  # word a是全局的第一个词时不需要修改前一个词的统计
                try:
                    self.word_pair_len[pre_word, word_a] -= 1
                except KeyError:
                    pass
                if pre_word == word_b and self.get_pre_word(pre_start) == word_a:  # 例如“1 0 1 0” 合并为 “10 10”的情况
                    new_pairs[word_comb, word_comb].append(pre_start - len_a)
                else:
                    new_pairs[pre_word, word_comb].append(pre_start)
        
            if nxt_word != "#":  # word b是全局的最后一个词时不需要修改后一个词的统计
                # 前后是 word_comb时，处理一侧即可
                if not (nxt_word == word_a and self.get_nxt_word(nxt_end) == word_b):
                    try:
                        self.word_pair_len[word_b, nxt_word] -= 1
                    except KeyError:
                        pass
                    new_pairs[word_comb, nxt_word].append(i)
        self.update(new_pairs)

        if len_b == 1:
            for i in indices:
                seg_status[i] = len_comb
                seg_status[i + len_a] = len_comb
        else:
            for i in indices:
                seg_status[i] = len_comb
                seg_status[i + len_a] = 0
                seg_status[i + len_comb - 1] = len_comb

        word_pair_v2.pop(comb)
        self.word_pair_len.pop(comb)
        
        # 如果要使用BPE的推理算法进行分词，则不能移除任何之前的合并结果
        # if self.word_count[word_a] <= 0:
        #     p = self.vocab.pop(word_a, None)
        # if word_b != word_a and self.word_count[word_b] <= 0:
        #     p = self.vocab.pop(word_b, None)
        

    def update(self, new_pairs):
        for k, v in new_pairs.items():
            if len(v) >= self.min_freq:
                data = self.word_pair_pos[k] = array("I", v)
                freq = self.word_pair_len[k] = len(data)
                heapq.heappush(self.pair_freq_queue, (-freq, k))

    def get_pre_word(self, init_pos):
        pos = init_pos - self.seg_status[init_pos - 1] 
        return self.corpus[pos: init_pos]

    def get_nxt_word(self, init_pos):
        pos = init_pos + self.seg_status[init_pos]
        return self.corpus[init_pos: pos]

    @staticmethod
    def replace_punc(text: str, verbose: bool = False):
        if verbose:
            print("Rmoving duplicate lines...")
        # 将text中所有中英文标点替换为 #号
        lines = text.split("\n")
        new_lines = set(lines)
        text = "#".join(lines)
        if verbose:
            print("lines:", len(lines), "new_lines:", len(new_lines), "ratio:", len(new_lines) / len(lines))
        del lines# , new_lines
        puncs_zh = [' ', '。', '，', '？', '！', '；', '：', '、', '（', '）', '「',
                    '」', '“', '”', '‘', '’', '《', '》', '【', '】', '…', '—', '～']
        puncs_en = ['.', ',', '?', '!', ';', ':', 
                    '(', ')', '"', '"', '\'', '\'', '<', '>', '[', ']', '.','~']
        puncs = (*puncs_zh, *puncs_en, "\n", "\t")
        i_name = sys.implementation.name
        if i_name == 'cpython':
            if verbose:
                print(f"[{i_name}] using string.translate for replacing punctuations")
            table = str.maketrans(dict.fromkeys(puncs, "#"))
            return text.translate(table)
        else:
            if verbose:
                print(f"[{i_name}] using re.sub for replacing punctuations")
            puncs = '|'.join(f"{re.escape(punc)}" for punc in puncs)
            pattern = re.compile(f"({puncs})+")
            return pattern.sub("#", text)

if __name__ == "__main__":
    a = BPETrainer(
        int(1e4), min_freq=15,
        compress_threshold=0.3,
        single_char=False
    ).train_from_file(
        "data/train_BPE.txt",
        verbose=True,
    )
    a.dump_vocab("output/epbe_vocab.json")
    with open("vocab.txt", "w") as f:
        f.write("\n".join(a.vocab.keys()))
    with open("data/test_BPE.txt") as f:
        tokens = a.tokenize(f.read())
    with open("tokenization_result.txt", "w") as f:
        f.write(" ".join(tokens))
