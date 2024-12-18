"""
Run analogy test using Google analogy test dataset
"""
from typing import List

import torch
import torch.nn.functional as F

import numpy as np

TOTAL_NUM_QUESTIONS = 19558

class AnalogyTest:
    """Analogy test class"""
    def __init__(self,
                 testset_src: str,
                 embed_file: str,
                 vocab_file: str,
                 max_num_questions: int):
        self.model = embed_file.split('_')[0]
        self.embeddings = torch.load(embed_file, weights_only=True).cpu()
        self.vocab = torch.load(vocab_file, weights_only=True)
        self.idx_to_word = {i: w for w, i in self.vocab.items()}
        self.questions = self.prepare_test(testset_src, max_num_questions)

    def prepare_test(self, test_src: str, max_questions: int):
        """Prepare testset from dataset"""
        np.random.seed(1234)
        ln_nums = set(np.random.choice(TOTAL_NUM_QUESTIONS, max_questions, replace=False))
        questions = []
        with open(test_src, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in ln_nums and not line.startswith(':'):
                        analogies = line.lower().split()
                        if all(w in self.vocab for w in analogies):
                            questions.append([self.vocab[w] for w in analogies])
        return questions

    def predict(self, question: List[int]) -> bool:
        """
        Predict function:
        v1, v2, u1, ? => v1 - v2 = u1 - ? => ? = u1 + v2 - v1
        """
        # map embeddings for each word in the question
        w1, w2, u1, answer = question
        V_w1, V_w2, V_u1 = self.embeddings[w1], self.embeddings[w2], self.embeddings[u1]

        # predict the most probable word
        V_d = V_u1 + V_w2 - V_w1
        cos_sims = F.cosine_similarity(self.embeddings, V_d.unsqueeze(0), dim=1)
        prediction = torch.argmax(cos_sims).item()

        # display the wrong prediction
        if answer != prediction:
            W_w1, W_w2, W_u1, W_ans = [self.idx_to_word[i] for i in question]
            W_pred = self.idx_to_word[prediction]
            print(f"{W_w1} : {W_w2} = {W_u1} : {W_pred} [ans: {W_ans}]")
            return False
        return True

    def score(self):
        """Score the embeddings on analogy test"""
        results = [self.predict(q) for q in self.questions]
        print(f"{'=' * 50}\nResult: {sum(results)}/{len(results)}")









