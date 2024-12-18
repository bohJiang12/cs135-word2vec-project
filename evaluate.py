"""
Main file for running evaluation on trained embeddings
"""
from eval.visualize import *
from eval.analogy import *
from eval.cos_sim import *
from eval.eval_utils import *

import warnings
from argparse import ArgumentParser

import torch

OUT_DIR = 'results'
WEIGHTS_DIR = 'weights'
DATA_DIR = 'data'


class Evaluator:
    """Class for evaluating trained embeddings on different datasets"""
    def __init__(self,
                 model: str,
                 dataset: int):
        """
        Args:
            model: the name of model training word2vec embeddings: either `cbow` or `sg`
            dataset: the indicator for choosing the dataset to evaluate on - 0: WordSim-353; 1: SimLex-999
        """
        self.model = model
        self.dataset = dataset

        # extended attributes from input arguments
        if self.model == 'cbow':
            self.embed_file = f"{WEIGHTS_DIR}/{self.model}_win4_d100_neg3_ep10.pt"
        else:
            self.embed_file = f"{WEIGHTS_DIR}/{self.model}_win4_d100_neg3_ep5.pt"
        self.vocab_file = f"{WEIGHTS_DIR}/{self.model}_vocab.pt"

    def visualize_embeddings(self):
        """Visualize both random/trained embeddings on given dataset"""
        v = Visualizer(self.embed_file, self.vocab_file, DATA_DIR, 1234, 50, self.dataset)
        v.plot(OUT_DIR)

    def analogy_test(self):
        """Analogy test on Google's analogy test dataset for around 150 questions"""
        test = AnalogyTest(testset_src=f"{DATA_DIR}/questions-words.txt",
                           embed_file=self.embed_file,
                           vocab_file=self.vocab_file,
                           max_num_questions=200)
        test.score(OUT_DIR)

    def cos_sim(self):
        """Compute cosine similarities for word pairs of a given dataset"""
        compute_sim(embed_file=self.embed_file, vocab_file=self.vocab_file, dataset=self.dataset, data_dir=DATA_DIR, out_dir=OUT_DIR)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="enter either 'cbow' or 'sg' for choosing CBOW or skip-gram respectively")
    parser.add_argument('-d', '--dataset', required=True, help="enter either 'wordsim' or 'simlex' for choosing WordSim-353 or SimLex-999 dataset")
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = get_args()
    model, dataset = args.model, args.dataset

    dataset_map = {'wordsim': 0, 'simlex': 1}

    e = Evaluator(model, dataset_map[dataset])

    print('Plotting embedding visualization ...')
    e.visualize_embeddings()
    print('done.\n')

    print('Evaluating analogy test ...')
    e.analogy_test()
    print('done.\n')

    print('Evaluating cosine similarities ...')
    e.cos_sim()
    print('done.\n')

    print(f"Check results under dir {OUT_DIR}/")







