import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CPGDatasetPackPadding(Dataset):
    def __init__(self, sequences, labels):
        """
        Initialize the dataset with sequences and labels (CpG counts) as input arguments and store them as attributes.

        Parameters:
        - sequences (List[List[int]]): List of integer-encoded DNA sequences.
        - labels (List[int]): List of CpG counts corresponding to each sequence.

        Returns:
        - None
        """
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a sequence and its label by index.
        """
        return self.sequences[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to pack sequences dynamically and return a batch of sequences and labels with lengths.
        """
        sequences, labels = zip(*batch)
        # find the lenghts of the sequences and sort them
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        sorted_indices = torch.argsort(lengths, descending=True)
        # sort sequences by length
        sequences = [sequences[i] for i in sorted_indices]
        # sort labels by length
        labels = torch.tensor([labels[i] for i in sorted_indices], dtype=torch.float32)
        lengths = lengths[sorted_indices]

        # pad sequences and return a packed sequence and labels and lengths
        pad_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

        return pad_sequences, labels, lengths
