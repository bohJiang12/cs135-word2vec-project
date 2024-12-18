"""
Utility functions for helping evaluation process
"""
from typing import Dict, Tuple

def load_simlex(fpath: str) -> Dict[Tuple[str, str], float]:
    """Function for loading SimLex-999 dataset"""
    pair_score = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 0:  # exclude header
                entries = line.split()
                w1, w2 = entries[0], entries[1]
                sim_val = float(entries[3])
                pair_score[(w1, w2)] = sim_val
    return pair_score


def load_wordsim(fpath: str) -> Dict[Tuple[str, str], float]:
    """Function for loading wordsim-353 dataset"""
    pair_score = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            entries = line.split()
            w1, w2 = entries[0], entries[1]
            sim_val = float(entries[2])
            pair_score[(w1, w2)] = sim_val
    return pair_score


def most_common_k(dataset: Dict[Tuple[str, str], float], k: int) -> Dict[Tuple[str, str], float]:
    """Find the most k similar word pairs from the dataset"""
    word_pairs = []
    count = 0
    for pair in sorted(dataset, key=dataset.get, reverse=True):
        if count < k:
            w1, w2 = pair
            if w1 != w2:
                word_pairs.append(pair)
                count += 1
        else:
            return word_pairs



