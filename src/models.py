"""
Module for implementing Word2Vec using algorithms:
- CBOW
- Skip-gram
"""
import torch
import torch.nn as nn

class CBOW(nn.Module):
    """CBOW model"""
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.W = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.C = nn.Embedding(vocab_size, embed_size, padding_idx=0)

    def forward(self, context, target, neg_samples):
        context_embed = self.W(context).mean(dim=1)
        target_embed = self.C(target)
        neg_samples_embed = self.C(neg_samples)

        # Compute positive and negative scores for loss function
        pos_score = torch.sum(context_embed * target_embed, dim=1)
        neg_score = torch.bmm(neg_samples_embed, context_embed.unsqueeze(2)).squeeze(2)
        
        # Concatenate logits and labels
        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        labels = torch.cat([
            torch.ones_like(pos_score.unsqueeze(1)), 
            torch.zeros_like(neg_score) 
        ], dim=1)

        return logits, labels



class SG(nn.Module):
    """Skip-gram model"""
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.W = nn.Embedding(vocab_size, embed_size)
        self.C = nn.Embedding(vocab_size, embed_size)

    def forward(self, target, context, neg_samples):
        target_embed = self.W(target)
        context_embed = self.C(context)
        neg_samples_embed = self.C(neg_samples)

        # Compute positive and negative scores for loss function
        pos_score = torch.sum(target_embed * context_embed, dim=1)
        neg_score = torch.bmm(neg_samples_embed, target_embed.unsqueeze(2)).squeeze(2)
        
        # Concatenate logits and labels
        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        labels = torch.cat([
            torch.ones(pos_score.shape[0], 1), 
            torch.zeros(pos_score.shape[0], neg_score.shape[1]) 
        ], dim=1)

        return logits, labels
     
    
    





