"""
Report (cosine) similarities for word pairs in chosen dataset
"""
from eval.eval_utils import *

import torch
import torch.nn.functional as F

SIMLEX_DATA = 'SimLex-999/SimLex-999.txt'
WORDSIM_DATA = 'wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'

def compute_sim(embed_file: str,
                vocab_file: str,
                dataset: int,
                data_dir: str,
                out_dir: str):
    """compute cosine similarities for word pairs in given dataset"""
    model = embed_file.split('/')[1].split('_')[0]
    embeddings = torch.load(embed_file, weights_only=True).cpu()
    vocab = torch.load(vocab_file, weights_only=True)
    if not dataset:
        data = load_simlex(f"{data_dir}/{SIMLEX_DATA}")
    data = load_wordsim(f"{data_dir}/{WORDSIM_DATA}")

    # compute cosine similarities and report them along with existing human ratings
    dataset_map = {0: 'wordsim', 1: 'simlex'}
    out_file = f"{out_dir}/{model}_{dataset_map[dataset]}_cos_sim.txt"
    with open(out_file, 'w') as f:
        header = "pair human_rating cos_similarity\n"
        f.write(header)
        for pair, rating in data.items():
            if all(w in vocab for w in pair):
                w1, w2 = pair
                v1, v2 = embeddings[vocab[w1]], embeddings[vocab[w2]]
                cos_sim = F.cosine_similarity(v1, v2, dim=0).item()
                line = f"{pair} {rating} {cos_sim:.2f}\n"
                f.write(line)





