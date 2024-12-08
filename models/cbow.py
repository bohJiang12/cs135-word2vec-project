import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm  # For progress bars

# Ensure imports work correctly
sys.path.append('../data')
from download_and_preprocess import preprocess, Vocabulary


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


def train_cbow_model(model, data_loader, num_epochs=5, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for context, target in pbar:
                optimizer.zero_grad()
                output = model(context)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(data_loader)}")


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    train_data = dataset["train"]
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
    model = CBOWModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
    train_cbow_model(model, data_loader, num_epochs=5, learning_rate=0.01)
