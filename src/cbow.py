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

# Ensure imports work correctly
sys.path.append('../data')
from download_and_preprocess import Vocabulary


nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words


def remove_stopwords(text):
    """Remove stopwords using spaCy."""
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered_words)



def preprocess_with_nltk(text):
    sentences = sent_tokenize(text)
    result = [remove_stopwords(sent) for sent in sentences]
    return result

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by half every 10 epochs

    losses = []  # List to track losses for each epoch
    start_time = time.time()  # Start tracking time

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

        scheduler.step()  # Update learning rate

    end_time = time.time()  # End tracking time
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Save and plot training loss
    os.makedirs("loss_plots", exist_ok=True)
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig("loss_plots/training_loss.png")
    plt.close()

    return model

def visualize_embeddings(embeddings, vocab, title, filename, num_words=50):
    """Visualize embeddings using TSNE."""
    reduced_embeddings = TSNE(n_components=2).fit_transform(embeddings[:num_words])
    words = [vocab.idx_to_word[i] for i in range(num_words)]

    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word, fontsize=8)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
def load_data(file_path):
    """Load raw data from a file."""
    print(f"Loading data from: {file_path}")
    with open(file_path, "r") as file:
        raw_data = file.readlines()
    print(f"Number of raw training samples: {len(raw_data)}")
    return raw_data


def preprocess_data(raw_data):
    """Preprocess raw data by tokenizing sentences and removing stopwords."""
    preprocessed_data = []
    for line in tqdm(raw_data, desc="Preprocessing data"):
        if line.strip():
            sentences = preprocess_with_nltk(line.lower())
            cleaned_sentences = [remove_stopwords(sentence) for sentence in sentences]
            preprocessed_data.extend(cleaned_sentences)
    print(f"Number of non-empty training samples: {len(preprocessed_data)}")
    print(f"Sample preprocessed data: {preprocessed_data[:10]}")
    return preprocessed_data


def build_vocabulary(preprocessed_data):
    """Build vocabulary from preprocessed data."""
    vocab = Vocabulary()
    for sentence in tqdm(preprocessed_data, desc="Building Vocabulary"):
        for word in sentence.split():
            vocab.add_word(word)
    print(f"Vocabulary size after stopwords removal: {len(vocab)}")
    return vocab


def tokenize_data(preprocessed_data, vocab):
    """Tokenize preprocessed data using the vocabulary."""
    tokenized_data = [vocab.encode(sentence) for sentence in tqdm(preprocessed_data, desc="Tokenizing data")]
    return tokenized_data


def filter_short_sentences(tokenized_data, window_size):
    """Filter sentences shorter than the required minimum length."""
    min_length = 2 * window_size + 1
    filtered_data = [tokens for tokens in tokenized_data if len(tokens) >= min_length]
    print(f"Number of sentences after filtering short ones: {len(filtered_data)}")
    return filtered_data


def compute_similarity(embeddings, vocab, word_pairs):
    """Compute cosine similarity for given word pairs."""
    for word1, word2 in word_pairs:
        try:
            vec1 = embeddings[vocab[word1]]
            vec2 = embeddings[vocab[word2]]
            similarity = cosine_similarity(vec1, vec2)
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")


def perform_analogies(embeddings, vocab, analogy_tasks):
    """Perform analogy tasks and print results."""
    for (word1, word2, word3) in analogy_tasks:
        try:
            vec1 = embeddings[vocab[word1]]
            vec2 = embeddings[vocab[word2]]
            vec3 = embeddings[vocab[word3]]
            analogy_vec = vec1 - vec2 + vec3
            similarities = np.dot(embeddings, analogy_vec) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(analogy_vec)
            )
            most_similar_idx = similarities.argsort()[-2]  # Exclude last input word
            most_similar_word = vocab.idx_to_word[most_similar_idx]
            print(f"{word1} - {word2} + {word3} = {most_similar_word}")
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")


def main():
    SEED = 42
    set_seed(SEED)

    # File path
    file_path = "../data/train.txt"
    raw_data = load_data(file_path)

    # Preprocessing
    preprocessed_data = preprocess_data(raw_data)

    # Vocabulary and tokenization
    vocab = build_vocabulary(preprocessed_data)
    tokenized_data = tokenize_data(preprocessed_data, vocab)

    # Generate training data
    window_size = 3
    filtered_data = filter_short_sentences(tokenized_data, window_size)
    cbow_data = generate_cbow_data(filtered_data, window_size)

    # DataLoader
    batch_size = 64
    data_loader = prepare_dataloader(cbow_data, batch_size)

    # Initialize and train CBOW model
    embedding_dim = 100
    num_epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CBOWModel(vocab_size=len(vocab), embedding_dim=embedding_dim).to(device)

    visualize_embeddings(
        model.embeddings.weight.detach().cpu().numpy(),
        vocab,
        "Embeddings Before Training",
        f"loss_plots/initial_embeddings{num_epochs}.png"
    )

    trained_model = train_cbow_model(model, data_loader, num_epochs, learning_rate=0.001, device=device)

    # Visualize embeddings after training
    visualize_embeddings(
        trained_model.embeddings.weight.detach().cpu().numpy(),
        vocab,
        "Embeddings After Training",
        f"loss_plots/trained_embeddings{num_epochs}.png"
    )

    # Word similarity tasks
    word_pairs = [
        ("team", "player"),
        ("doctor", "nurse"),
        ("car", "truck"),
        ("happy", "joyful"),
        ("hot", "cold")
    ]
    compute_similarity(trained_model.embeddings.weight.detach().cpu().numpy(), vocab, word_pairs)

    # Analogy tasks
    analogy_tasks = [
        ("king", "man", "woman"),
        ("paris", "france", "italy"),
        ("king", "kings", "queen")
    ]
    perform_analogies(trained_model.embeddings.weight.detach().cpu().numpy(), vocab, analogy_tasks)


if __name__ == "__main__":
    main()
