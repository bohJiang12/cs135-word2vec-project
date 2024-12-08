import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


class Vocabulary:
    def __init__(self):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}  # Initialize with special tokens
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}

    def add_word(self, word):
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def __len__(self):
        return len(self.word_to_idx)

    def encode(self, text):
        return [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in text.split()]

    def decode(self, indices):
        return [self.idx_to_word.get(idx, "<unk>") for idx in indices]
    
    def __getitem__(self, word):
        return self.word_to_idx.get(word, None)  # Returns None if the word is not in vocabulary


class WikiTextDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, dtype=torch.long)


def preprocess(text):
    """
    Preprocesses the input text by converting to lowercase and stripping whitespace.
    Removes empty or invalid entries.
    """
    return text.lower().strip()


def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)  # 0 for <pad>


def main():
    # Initialize Vocabulary
    vocab = Vocabulary()

    # Load the WikiText-103 dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    train_data = dataset["train"]
    print(f"Number of raw training samples: {len(train_data)}")

    # Preprocess and build vocabulary
    train_data = [preprocess(item["text"]) for item in train_data if item["text"].strip()]
    print(f"Number of non-empty training samples: {len(train_data)}")

    for text in train_data:
        for word in text.split():
            vocab.add_word(word)

    print(f"Vocabulary size: {len(vocab)}")

    # Create the dataset and DataLoader
    wiki_dataset = WikiTextDataset(train_data, vocab)
    data_loader = DataLoader(wiki_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Iterate over the DataLoader
    print("\nSample batch of tokenized data (padded):")
    for batch in data_loader:
        print(batch)
        print("Batch shape:", batch.shape)  # Confirm all sequences are of equal length
        break


if __name__ == "__main__":
    main()
