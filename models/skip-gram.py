import sys
import os
import time  # For tracking training time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import random
from cbow import visualize_embeddings, remove_stopwords, preprocess_with_nltk, set_seed

# Ensure imports work correctly
sys.path.append('../data')
from download_and_preprocess import Vocabulary


nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words




class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word):
        embedded = self.embeddings(target_word)  # [batch_size, embedding_dim]
        output = self.linear(embedded)  # [batch_size, vocab_size]
        return output


class SkipGramDataset(Dataset):
    def __init__(self, skipgram_data):
        self.data = skipgram_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

def prepare_dataloader(cbow_data, batch_size):
    dataset = SkipGramDataset(cbow_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def generate_skipgram_data(tokenized_data, window_size=2):
    skipgram_data = []
    for sentence in tqdm(tokenized_data, desc="Generating Skip-Gram data"):
        for i in range(len(sentence)):
            for j in range(1, window_size + 1):
                if i - j >= 0:
                    skipgram_data.append((sentence[i], sentence[i - j]))
                if i + j < len(sentence):
                    skipgram_data.append((sentence[i], sentence[i + j]))
    return skipgram_data


def train_skipgram_model(model, data_loader, num_epochs, learning_rate=0.01, device=torch.device("cpu")):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for target, context in pbar:
                target = target.to(device)
                context = context.to(device)
                optimizer.zero_grad()
                output = model(target)
                loss = criterion(output, context)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        scheduler.step()

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig(f"loss_plots/skipgram_training_loss{num_epochs}.png")
    plt.close()

    return model


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    # Load the dataset from the file
    file_path = "../data/train.txt"
    print(f"Loading data from: {file_path}")
    with open(file_path, "r") as file:
        raw_data = file.readlines()

    print(f"Number of raw training samples: {len(raw_data)}")

    # Preprocess data with stopwords removal
    preprocessed_data = []
    for line in tqdm(raw_data, desc="Preprocessing data"):
        if line.strip():
            # Process each line into sentences, remove stopwords from each sentence
            sentences = preprocess_with_nltk(line.lower())
            cleaned_sentences = [remove_stopwords(sentence) for sentence in sentences]
            preprocessed_data.extend(cleaned_sentences)

    print(f"Number of non-empty training samples: {len(preprocessed_data)}")
    print(f"Sample preprocessed data: {preprocessed_data[:10]}")


    # Build vocabulary
    vocab = Vocabulary()
    for sentence in tqdm(preprocessed_data, desc="Building Vocabulary"):
        for word in sentence.split():
            vocab.add_word(word)
    print(f"Vocabulary size after stopwords removal: {len(vocab)}")

    # Tokenize data
    tokenized_data = [vocab.encode(sentence) for sentence in tqdm(preprocessed_data, desc="Tokenizing data")]

    # Generate CBOW data
    window_size = 3
    min_length = 2 * window_size + 1
    tokenized_data = [tokens for tokens in tokenized_data if len(tokens) >= min_length]
    print(f"Number of sentences after filtering short ones: {len(tokenized_data)}")

    # Generate Skip-Gram data
    skipgram_data = generate_skipgram_data(tokenized_data, window_size)

    # Prepare DataLoader
    batch_size = 64
    data_loader = prepare_dataloader(skipgram_data, batch_size)

    # Initialize and train Skip-Gram model
    embedding_dim = 100
    num_epochs = 35
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SkipGramModel(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)

    visualize_embeddings(
        model.embeddings.weight.detach().cpu().numpy(),
        vocab,
        "Embeddings Before Training",
        f"loss_plots/skipgram_initial_embeddings{num_epochs}.png"
    )

    trained_model = train_skipgram_model(model, data_loader, num_epochs, learning_rate=0.001, device=device)

    # Visualize embeddings after training
    visualize_embeddings(
        trained_model.embeddings.weight.detach().cpu().numpy(),
        vocab,
        "Embeddings After Training",
        f"loss_plots/skipgram_trained_embeddings{num_epochs}.png"
    )
