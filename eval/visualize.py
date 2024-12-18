"""
Visualize embeddings
"""
from eval.eval_utils import *

from typing import List

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import numpy as np

from umap import UMAP

SIMLEX_DATA = 'SimLex-999/SimLex-999.txt'
WORDSIM_DATA = 'wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'

class Visualizer:
    """Word embedding visualizer using UMAP"""
    def __init__(self,
                 embed_file: str,
                 vocab_file: str,
                 data_dir: str,
                 seed: int,
                 num_words: int,
                 dataset: int):
        """
        Note:
        `dataset`: an indicator for choosing dataset - 0 for SimLex-999; 1 for wordsim-353
        """
        self.model = embed_file.split('/')[1].split('_')[0]
        self.embeddings = torch.load(embed_file, weights_only=True)
        self.vocab = torch.load(vocab_file, weights_only=True)

        self.idx_to_word = {i: w for w, i in self.vocab.items()}
        self.seed = seed
        self.k = num_words
        self.dataset = dataset
        self.data_dir = data_dir


    def _select_words(self) -> List[int]:
        """select top k most similar pairs of words as indices"""
        if self.dataset:
            word_pairs = most_common_k(load_wordsim(f"{self.data_dir}/{WORDSIM_DATA}"), self.k)
        else:
            word_pairs = most_common_k(load_simlex(f"{self.data_dir}/{SIMLEX_DATA}"), self.k)

        words = []
        for pair in word_pairs:
            if all(word in self.vocab for word in pair):
                w1, w2 = pair
                words.append(self.vocab[w1])
                words.append(self.vocab[w2])
        return words


    def _fit(self):
        # set random seed
        torch.manual_seed(self.seed)

        # init shape and dimension reducer
        V, d = self.embeddings.shape
        reducer = UMAP(n_components=2, random_state=self.seed)

        # select words
        self.word_indices = self._select_words()
        self.words = [self.idx_to_word[i] for i in self.word_indices]

        # fit random embeddings
        rand_embedding_layer = nn.Embedding(V, d)
        rand_embeddings = rand_embedding_layer(torch.tensor(self.word_indices, dtype=torch.long))
        self.rand_embed_2d = reducer.fit_transform(rand_embeddings.detach().numpy())

        # fit trained embeddings
        selected_embeddings = self.embeddings[self.word_indices]
        self.trained_embed_2d = reducer.fit_transform(selected_embeddings.cpu().numpy())

    def plot(self, out_dir: str):
        self._fit()

        # create a figure for plotting two embeddings
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # plot the random embeddings first
        axes[0].scatter(self.rand_embed_2d[:, 0], self.rand_embed_2d[:, 1], alpha=0.7, label='random embeddings')
        for i, word in enumerate(self.words):
            axes[0].text(self.rand_embed_2d[i, 0], self.rand_embed_2d[i, 1], word, fontsize=8, alpha=0.7)
        axes[0].set_title('Random embeddings')
        axes[0].set_xlabel('$d_1$')
        axes[0].set_ylabel('$d_2$')
        axes[0].legend()

        # plot the trained embeddings first
        axes[1].scatter(self.trained_embed_2d[:, 0], self.trained_embed_2d[:, 1], alpha=0.7, label='trained embeddings')
        for i, word in enumerate(self.words):
            axes[1].text(self.trained_embed_2d[i, 0], self.trained_embed_2d[i, 1], word, fontsize=8, alpha=0.7)
        axes[1].set_title('Trained embeddings')
        axes[1].set_xlabel('$d_1$')
        axes[1].set_ylabel('$d_2$')
        axes[1].legend()

        plt.tight_layout()

        dataset_map = {0: 'wordsim', 1: 'simlex'}
        plt.savefig(f"{out_dir}/{self.model}_{dataset_map[self.dataset]}_scatter.png", dpi=300, bbox_inches='tight')










