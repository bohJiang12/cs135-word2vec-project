"""
Main file for training models and save them
"""

from src.models import CBOW, SG
from src.dataset import CBOWDataset, SGDataset
from src.utils import Config

import os
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np

def cbow_collate_fn(batch, vocab_size, num_negs, words_freq):
    contexts, targets = zip(*batch)
    
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    
    neg_samples = []
    for _ in range(len(targets)):
        neg_words = np.random.choice(
            vocab_size, size=num_negs, replace=False, p=words_freq
        )
        neg_samples.append(neg_words)
    neg_samples = np.array(neg_samples)
    neg_samples = torch.tensor(neg_samples, dtype=torch.long)
    return contexts, targets, neg_samples


def sg_collate_fn(batch, vocab_size, num_negs, words_freq):
    targets, contexts = zip(*batch)  # key difference with CBOW
    
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    
    neg_samples = []
    for _ in range(len(targets)):
        neg_words = np.random.choice(
            vocab_size, size=num_negs, replace=False, p=words_freq
        )
        neg_samples.append(neg_words)
    neg_samples = np.array(neg_samples)
    neg_samples = torch.tensor(neg_samples, dtype=torch.long)
    return targets, contexts, neg_samples


class Run:
    """Class for training word2vec model"""
    def __init__(self, config: Config):
        self.params = config.params
        self.device = torch.device(self.params['device'])
    
    def _set_up(self):
        """Set up dataloader, optimizer and loss func for training"""
        # Dataloader
        train_data = f"{self.params['data_dir']}/{self.params['corpus']}"
        if self.params['model'] == 'cbow':
            dataset = CBOWDataset(train_data, self.params['win_size'])
            self.vocab = dataset.vocab
            dist = dataset.dist
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.params['batch_size'],
                shuffle=True,
                collate_fn=lambda batch: cbow_collate_fn(batch, len(self.vocab), self.params['n_neg_sps'], dist)
                )
            self.model = CBOW(len(dataset.vocab), self.params['embed_dim']).to(self.device)
        else:
            dataset = SGDataset(train_data, self.params['win_size'])
            self.vocab = dataset.vocab
            dist = dataset.dist
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.params['batch_size'],
                shuffle=True,
                collate_fn=lambda batch: sg_collate_fn(batch, len(self.vocab), self.params['n_neg_sps'], dist)
                )
            self.model = SG(len(self.vocab), self.params['embed_dim']).to(self.device)

        # Optimizer and loss function
        if self.params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'])
        
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def train(self):
        self._set_up()
        if self.params['model'] == 'cbow':
            self.train_cbow()
        else:
            self.train_sg()
    
    def train_cbow(self):
        for epoch in range(self.params['epochs']):
            total_loss = 0.0
            with tqdm(total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.params['epochs']}", unit="batch") as pbar:
                for contexts, target, neg_samples in self.dataloader:
                    contexts, target, neg_samples = contexts.to(self.device), target.to(self.device), neg_samples.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    logits, labels = self.model(contexts, target, neg_samples)
                    loss = self.loss_fn(logits.to(self.device), labels.to(self.device))
                    
                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix(loss=total_loss / (pbar.n + 1))  # Show average loss
                    pbar.update(1)
            print(f"Epoch {epoch + 1}/{self.params['epochs']} | Loss: {total_loss:.4f}")
            
    def train_sg(self):
        for epoch in range(self.params['epochs']):
            total_loss = 0.0
            with tqdm(total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.params['epochs']}", unit="batch") as pbar:
                for target, context, neg_samples in self.dataloader:
                    context, target, neg_samples = context.to(self.device), target.to(self.device), neg_samples.to(self.device)
        
                    self.optimizer.zero_grad()
                    
                    logits, labels = self.model(target, context, neg_samples)
                    loss = self.loss_fn(logits.to(self.device), labels.to(self.device))
                    
                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    pbar.set_postfix(loss=total_loss / (pbar.n + 1))  # Show average loss
                    pbar.update(1)
            print(f"Epoch {epoch + 1}/{self.params['epochs']} | Loss: {total_loss:.4f}")
    
    def save_embeddings(self):
        out_dir = self.params['out_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # embeddings = self.model.W.weight.data.cpu().numpy()
        # with open(f"{out_dir}/{filename}.txt", 'w') as f:
        #     for word, idx in self.vocab.items():
        #         Vw = embeddings[idx]
        #         Vw_str = " ".join(map(str, Vw))
        #         f.write(f"{word} {Vw_str}\n")
        embed_name = f"{self.params['model']}_win{self.params['win_size']}_d{self.params['embed_dim']}_neg{self.params['n_neg_sps']}_ep{self.params['epochs']}.pt"
        torch.save(self.model.W.weight.data, f"{out_dir}/{embed_name}")
        torch.save(self.vocab, f"{out_dir}/{self.params['model']}_vocab.pt")
        
        
def train(config: Config):
    """Function for training word2vec model with external config file"""
    params = config.params
    device = torch.device(params['device'])
    
    # Dataloader
    train_data = f"{params['data_dir']}/{params['corpus']}"
    if params['model'] == 'cbow':
        dataset = CBOWDataset(train_data, params['win_size'], params['n_neg_sps'])
        dataloader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=True
            )
        model = CBOW(len(dataset.vocab), params['embed_dim']).to(device)
    else:
        raise ValueError("Skip-gram hasn't been implemented yet!")

    # Optimizer and loss function
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Start training in epochs
    prev_loss = 0.0
    round_count = 0
    tol, rounds = params['tol'], params['rounds']
    for epoch in range(params['epochs']):
        total_loss = 0.0
        for contexts, target, neg_samples in dataloader:
            contexts, target, neg_samples = contexts.to(device), target.to(device), neg_samples.to(device)
            
            context_size = contexts.size(0)
            pos_labels = torch.ones(context_size).to(device)
            neg_labels = torch.zeros(context_size, params['n_neg_sps']).to(device)
            
            optimizer.zero_grad()
            
            pos_score, neg_score = model(contexts, target, neg_samples)
            pos_loss = loss_fn(pos_score, pos_labels)
            neg_loss = loss_fn(neg_score, neg_labels)
            loss = pos_loss + neg_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{params['epochs']}  | Loss: {total_loss:.4f}")
        
        # Early stop
        if round_count == rounds:
            print('Early stopped!')
            break
        if total_loss < prev_loss:
            if (prev_loss - total_loss) / prev_loss < tol:
                round_count += 1
        else:
            round_count += 1
        
    
    # TODO: add function for saving embeddings


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--configFile', required=True, help='Configuration file as YAML format')
    return parser.parse_args()

    
if __name__ == '__main__':
    args = get_args()
    run = Run(Config(args.configFile))
    
    print(f"{'=' * 25} START {'=' * 25}\n")
    run.train()
    print(f"{'=' * 25} END {'=' * 25}\n")
    
    print('Saving embeddings and vocabulary..')
    run.save_embeddings()
    print('Done!')
    