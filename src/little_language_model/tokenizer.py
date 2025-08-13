import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length=256, stride=128):
        """
        Initializes the dataset with a text and tokenizer.
        Reads the file line by line and tokenizes each line.
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns the input and target tensors for a given index.
        """
        return self.input_ids[idx], self.target_ids[idx]

def data_loader(file_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Loads data from a file and returns a DataLoader object.
    """
    with open(file_path, 'r') as txt:
        file_contents = txt.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataset(file_contents, tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        return dataloader
