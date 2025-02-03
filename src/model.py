import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CpGCounterAdvancedPackPadding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        """
        Initialize the CpGCounterAdvancedPackPadding model with an embedding layer, LSTM layer, batch normalization, and fully connected layer with ReLU activation.

        Parameters:
        - vocab_size (int): Number of unique tokens in the vocabulary.
        - embedding_dim (int): Dimension of the embedding layer.
        - hidden_size (int): Dimension of the LSTM hidden state.
        - num_layers (int): Number of LSTM layers.
        - dropout (float): Dropout rate.

        Returns:
        - None
        """

        super(CpGCounterAdvancedPackPadding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        """
        Forward pass using packed sequences to handle variable length sequences.

        Parameters:
        - x: (batch_size, seq_len) -> Padded sequences.
        - lengths: (batch_size) -> Actual sequence lengths.

        Returns:
        - CpG count prediction.
        """
        assert (
            torch.max(x).item() < self.embedding.num_embeddings
        ), "Index out of range!"
        embedded = self.embedding(
            x
        )  # input: (batch_size, seq_len), output: (batch_size, seq_len, embedding_dim)

        # Pack sequence to ignore padding
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM processing
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack sequence to get the output of each time step (hidden states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Extract last valid hidden state dynamically (from both directions)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_indices = lengths - 1
        last_hidden_states = output[batch_indices, last_indices, :]

        # Apply batch norm & fully connected layer
        last_hidden_states = self.batch_norm(last_hidden_states)
        output = self.fc(last_hidden_states)
        return self.relu(output)
