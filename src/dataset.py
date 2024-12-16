"""
Module for preparing dataset from corpus
"""
from abc import ABC, abstractmethod
from typing import Union, Iterator, Tuple, List, Dict, Set
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset

# from nltk.corpus import stopwords
#STOP_WORDS = set(stopwords.words('english'))

OOV = '<oov>'  # special token for out-of-vocabulary word

class Corpus:
    """Corpus class for collecting words from a raw corpus"""
    def __init__(self, fpath: Union[str, Path]):
        self.fpath = fpath
        self.words_count = Counter()
        self.sentences = self._load_corpus()

    def _load_corpus(self) -> Iterator[List[str]]:
        """Yield every sentence in the corpus and count each word"""
        with open(self.fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line and line != '"' and not line.startswith('='):
                    sentence = line.lower().split()
                    self.words_count.update(sentence)
                    yield sentence

class Data(Dataset, ABC):
    """Abstract dataset"""
    def __init__(self, corpus: Corpus, num_neg_samples: int):
        self.vocab = {w: i for i, w in enumerate(corpus.words_count)}
        self.vocab.update({OOV: len(self.vocab)})
        self.neg_samples = self._neg_sampling(corpus.words_count, num_neg_samples)
        self.data = None

    def __len__(self) -> int:
        assert self.data is not None
        return len(self.data)

    def __getitem__(self, index):
        assert self.data is not None
        contexts, target = self.data[index]
        return torch.tensor(contexts, dtype=torch.long), torch.tensor(target, dtype=torch.long), self.neg_samples

    def _neg_sampling(self, word_freq: Counter[str, int], n: int) -> torch.LongTensor:
        word_dist = torch.tensor(
            [word_freq[i] ** 0.75 for i in range(self.vocab)]
        )
        word_dist /= word_dist.sum()
        return torch.multinomial(word_dist, n, replacement=True)

    @abstractmethod
    def build_data(self):
        pass

class CBOWDataset(Data):
    """CBOW dataset"""
    def __init__(self, corpus: Corpus, num_neg_samples: int, win_size: int):
        super().__init__(corpus, num_neg_samples)
        self.data = self.build_data(corpus.sentences, win_size)

    def build_data(self, corpus: Iterator[List[str]], win_size: int) -> List[Tuple[List[int], int]]:
        pairs = []
        for sentence in corpus:
            for i, token in enumerate(sentence):
                start = max(0, i - win_size)
                end = min (i + win_size + 1, len(sentence))
                contexts = [self.vocab.get(w, self.vocab[OOV]) for w in sentence[start:i] + sentence[i+1:end]]
                target = self.vocab.get(token, self.vocab[OOV])
                pairs.append((contexts, target))
        return pairs

class SGDataset(Data):
    """Skip-gram dataset"""
    def __init__(self, corpus: Corpus, num_neg_samples: int, win_size: int):
        super().__init__(corpus, num_neg_samples)

    def build_data(self, corpus: Iterator[List[str]], win_size: int) -> List[Tuple[List[int], int]]:
        ...
























