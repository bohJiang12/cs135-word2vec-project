"""
Module for preparing dataset from corpus
"""
from abc import ABC, abstractmethod
from typing import Union, Iterator, Tuple, List, Dict, Set
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset

import numpy as np

# from nltk.corpus import stopwords
#STOP_WORDS = set(stopwords.words('english'))

OOV = '<oov>'  # special token for out-of-vocabulary word
PAD = '<pad>'  # special token for padding 

def get_dist(word_freq: Union[Dict, Counter], vocab: Dict[str, int]) -> torch.LongTensor:
    word_dist = np.array(
        [word_freq[w] ** 0.75 for w in vocab]
    )
    word_dist /= word_dist.sum()
    return word_dist


class Corpus:
    """Corpus class for collecting words from a raw corpus"""
    def __init__(self):
        self.words_count = Counter()

    def load_from(self, fpath: Union[str, Path]) -> Iterator[List[str]]:
        """Yield every sentence in the corpus and count each word"""
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line != '"' and not line.startswith('='):
                    sentence = line.lower().split()
                    self.words_count.update(sentence)
                    yield sentence

class CBOWDataset(Dataset):
    """CBOW dataset"""
    def __init__(self,
                 corpus_file: Union[str, Path],
                 win_size: int):
        self.data, self.vocab, self.dist = self.build_data(corpus_file, win_size)
        
    def __len__(self) -> int:
        assert self.data 
        return len(self.data)

    def __getitem__(self, index):
        assert self.data 
        contexts, target = self.data[index]
        contexts_indices = torch.tensor(
            [self.vocab.get(c, self.vocab[OOV]) for c in contexts],
            dtype=torch.long
        )
        target_index = torch.tensor(self.vocab.get(target, self.vocab[OOV]), dtype=torch.long)
    
        return contexts_indices, target_index 

    def build_data(self,
                   corpus_file: Union[str, Path],
                   win_size: int
                   ) -> Tuple[
                       Tuple[List[str], str],
                       Dict[str, int],
                       torch.LongTensor
                   ]:
        """
        Build pairs (contexual words, target) from given corpus file;
        Build vocabulary and pick negative samples from yielded word counter
        """
        # load corpus from file
        corpus = Corpus()
        sentences = corpus.load_from(corpus_file)
        
        # build pairs: (contexts, target)
        pairs = []
        for sentence in sentences:
            for i, target in enumerate(sentence):
                start = max(0, i - win_size)
                end = min(i + win_size + 1, len(sentence))
                contexts = sentence[start:i] + sentence[i+1:end]
                if len(contexts) < 2 * win_size:
                    contexts += (2*win_size - len(contexts)) * [PAD]
                pairs.append((contexts, target))
        
        # build vocab and negative samples based on words frequence
        vocab = {w: i+1 for i, w in enumerate(corpus.words_count)}
        vocab[PAD] = 0
        vocab.update({OOV: len(vocab)})
        
        word_dist = get_dist(corpus.words_count, vocab)
        
        return pairs, vocab, word_dist  


class SGDataset(CBOWDataset):
    """Skip-gram dataset"""
    def __init__(self,
                 corpus_file: Union[str, Path],
                 win_size: int):
        self.data, self.vocab, self.dist = self.build_data(corpus_file, win_size)
        
    def __len__(self) -> int:
        assert self.data 
        return len(self.data)

    def __getitem__(self, index):
        assert self.data 
        target, context = self.data[index]
        context_idx = torch.tensor(self.vocab.get(context, self.vocab[OOV]), dtype=torch.long)
        target_index = torch.tensor(self.vocab.get(target, self.vocab[OOV]), dtype=torch.long)
    
        return target_index, context_idx

    def build_data(self,
                   corpus_file: Union[str, Path],
                   win_size: int
                   ) -> Tuple[
                       List[Tuple[str, str]],
                       Dict[str, int],
                       torch.LongTensor
                   ]:
        """
        Build pairs (contexual words, target) from given corpus file;
        Build vocabulary and pick negative samples from yielded word counter
        """
        # load corpus from file
        corpus = Corpus()
        sentences = corpus.load_from(corpus_file)
        
        # build pairs: (target, context)
        pairs = []
        for sentence in sentences:
            for i, target in enumerate(sentence):
                start = max(0, i - win_size)
                end = min(i + win_size + 1, len(sentence))
                contexts = sentence[start:i] + sentence[i+1:end]
                for context in contexts:
                    pairs.append((target, context))
        
        # build vocab and negative samples based on words frequence
        vocab = {w: i for i, w in enumerate(corpus.words_count)}
        vocab.update({OOV: len(vocab)})
        
        word_dist = get_dist(corpus.words_count, vocab)
        return pairs, vocab, word_dist
            
























