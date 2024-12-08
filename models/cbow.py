import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt  # For plotting loss
from datasets import load_dataset

# Ensure imports work correctly
sys.path.append('../data')
from download_and_preprocess import preprocess, Vocabulary

# Set Hugging Face cache directory
os.environ["HF_DATASETS_CACHE"] = "/home/egk265/huggingface"


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)  # [batch_size, context_size, embedding_dim]
        context_vector = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        output = self.linear(context_vector)  # [batch_size, vocab_size]
        return output


class CBOWDataset(Dataset):
    def __init__(self, cbow_data):
        self.data = cbow_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def generate_cbow_data(tokenized_data, window_size=2):
    cbow_data = []
    for sentence in tqdm(tokenized_data, desc="Generating CBOW data"):
        for i in range(window_size, len(sentence) - window_size):
            context = sentence[i - window_size:i] + sentence[i + 1:i + 1 + window_size]
            target = sentence[i]
            cbow_data.append((context, target))
    return cbow_data


def prepare_dataloader(cbow_data, batch_size):
    dataset = CBOWDataset(cbow_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def train_cbow_model(model, data_loader, num_epochs, learning_rate=0.01, device=torch.device("cpu")):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []  # List to track losses for each epoch

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for context, target in pbar:
                # Move data to the GPU
                context = context.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(context)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)  # Track the average loss for this epoch
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # Save and plot training loss
    os.makedirs("loss_plots", exist_ok=True)
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig("loss_plots/training_loss.png")
    plt.close()

    return model


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir="/home/egk265/huggingface")
    subset_fraction = 0.01  # Use a subset of the dataset for testing
    train_data = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * subset_fraction)))

    print(f"Number of raw training samples: {len(train_data)}")

    # Preprocess data
    preprocessed_data = [
        preprocess(item["text"]) for item in tqdm(train_data, desc="Preprocessing data") if item["text"].strip()
    ]

    # Build vocabulary
    vocab = Vocabulary()
    for sentence in tqdm(preprocessed_data, desc="Building Vocabulary"):
        for word in sentence.split():
            vocab.add_word(word)
    print(f"Vocabulary size: {len(vocab)}")

    # Tokenize data
    tokenized_data = [vocab.encode(sentence) for sentence in tqdm(preprocessed_data, desc="Tokenizing data")]

    # Generate CBOW data
    window_size = 2
    cbow_data = generate_cbow_data(tokenized_data, window_size)

    # Prepare DataLoader
    batch_size = 32
    data_loader = prepare_dataloader(cbow_data, batch_size)

    # Initialize and train CBOW model
    embedding_dim = 100
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CBOWModel(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)
    trained_model = train_cbow_model(model, data_loader, num_epochs, learning_rate=0.01, device=device)

    # Compute similarity between words
    embeddings = trained_model.embeddings.weight.detach().cpu().numpy()
    try:
        king_vec = embeddings[vocab["king"]]
        queen_vec = embeddings[vocab["queen"]]
        similarity = cosine_similarity(king_vec, queen_vec)
        print(f"Similarity between 'king' and 'queen': {similarity:.4f}")
    except KeyError as e:
        print(f"Word not in vocabulary: {e}")
