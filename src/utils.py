import os
import torch
from src.model import CpGCounterAdvancedPackPadding
from typing import Sequence
from functools import partial
import random
import numpy as np
import random
import os
from typing import List, Tuple

# Alphabet helpers
alphabet = "NACGT"
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})  # Padding token
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)


def predict_cpgs_from_dna_pack_padded(
    model_path: str,
    dna_sequence: str,
    dna2int: dict,
    embedding_dim,
    hidden_size,
    num_layers,
    dropout,
    device,
    model_class=CpGCounterAdvancedPackPadding,  # Ensure the correct model class is used
):
    """
    Predict CpG count from a human DNA string.

    Parameters:
    - model_path: Path to trained LSTM model.
    - dna_sequence: Human-readable DNA string.
    - dna2int: Dictionary mapping DNA bases to integer values.
    - embedding_dim: Dimension of embedding layer.
    - hidden_size: Size of LSTM hidden state.
    - num_layers: Number of LSTM layers.
    - dropout: Dropout rate.
    - device: The device ('cpu' or 'cuda') for inference.
    - model_class: The model class to initialize the architecture.

    Returns:
    - Predicted CpG count (rounded to 2 decimal places).
    """

    # Check if the model checkpoint exists
    model_path = os.path.join("models", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load Model
    vocab_size = len(dna2int)
    model = model_class(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Load the trained model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)  # Move model to the correct device
    model.eval()

    # Convert DNA string to integer sequence
    int_sequence = [
        dna2int.get(base, 0) for base in dna_sequence
    ]  # Map bases to integers
    int_tensor = (
        torch.tensor(int_sequence, dtype=torch.long).unsqueeze(0).to(device)
    )  # Add batch dim

    # Compute sequence length (as tensor) and move to the same device
    lengths = torch.tensor([len(int_sequence)], dtype=torch.long).to(device)

    # Inference
    with torch.no_grad():
        predicted_count = (
            model(int_tensor, lengths).squeeze().item()
        )  # Ensure it's a scalar

    return round(predicted_count, 2)


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    save_path="best_cpg_model.pth",
    model_path="models",
):
    """
    Saves the model, optimizer, and scheduler state for training resumption.

    Parameters:
    - epoch (int): Current epoch number.
    - model (nn.Module): PyTorch model to save.
    - optimizer (torch.optim.Optimizer): Optimizer to save.
    - scheduler (torch.optim.lr_scheduler): Scheduler to save.
    - best_val_loss (float): Best validation loss achieved.
    - save_path (str): Path to save the checkpoint.

    """
    checkpoint = {
        "epoch": epoch + 1,  # Save next epoch to resume correctly
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    save_path = os.path.join(model_path, save_path)
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved at {save_path}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    device,
    save_path="best_cpg_model.pth",
    model_path="models",
):
    """
    Loads a saved checkpoint to resume training.

    Parameters:
    - model (nn.Module): PyTorch model to load the state_dict into.
    - optimizer (torch.optim.Optimizer): Optimizer to load the state_dict into.
    - scheduler (torch.optim.lr_scheduler): Scheduler to load the state_dict into.
    - device (torch.device): CPU or GPU.
    - save_path (str): Path to the saved checkpoint.

    """
    save_path = os.path.join(model_path, save_path)
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Loaded checkpoint from {save_path}, resuming from epoch {checkpoint['epoch']} with best validation loss: {best_val_loss:.4f}"
    )
    return model, optimizer, scheduler, checkpoint["epoch"], best_val_loss


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Generate fixed-length DNA sequences
def rand_sequence(n_seqs: int, seq_len: int = 128) -> List[List[int]]:
    return [[random.randint(1, 5) for _ in range(seq_len)] for _ in range(n_seqs)]


# Generate variable-length DNA sequences
def rand_sequence_var_len(n_seqs: int, lb: int = 16, ub: int = 128) -> List[List[int]]:
    return [
        [random.randint(1, 5) for _ in range(random.randint(lb, ub))]
        for _ in range(n_seqs)
    ]


# Count CpG sites in DNA sequence
def count_cpgs(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] == "CG")


# Function to prepare data
def prepare_data(
    mode: str, num_samples=100, min_len=16, max_len=128
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates DNA sequences and their corresponding CpG site counts.

    Parameters:
    - mode (str): Either "fixed" for fixed-length sequences or "variable" for variable-length sequences.
    - num_samples (int): Number of sequences to generate.
    - min_len (int): Minimum length of variable-length sequences (used if mode="variable").
    - max_len (int): Maximum length of variable-length sequences (used if mode="variable").

    Returns:
    - X_dna_seqs_train (List[List[int]]): List of integer-encoded DNA sequences.
    - y_dna_seqs (List[int]): List of CpG site counts corresponding to each sequence.
    """

    if mode == "fixed":
        X_dna_seqs_train = rand_sequence(num_samples, max_len)
    elif mode == "variable":
        X_dna_seqs_train = rand_sequence_var_len(num_samples, min_len, max_len)
    else:
        raise ValueError("Invalid mode. Choose 'fixed' or 'variable'.")

    # Convert integer sequences to DNA string sequences
    dna_sequences = ["".join(list(intseq_to_dnaseq(seq))) for seq in X_dna_seqs_train]

    # Count CpG sites
    y_dna_seqs = [count_cpgs(seq) for seq in dna_sequences]

    return X_dna_seqs_train, y_dna_seqs


def create_test_data(min_len=16, max_len=128) -> Tuple[str, int]:
    """
    Generates a single DNA sequence and its corresponding CpG site count.
    """
    X_dna_seqs_train = list(rand_sequence_var_len(1, min_len, max_len))
    test_dna = ["".join(list(intseq_to_dnaseq(seq))) for seq in X_dna_seqs_train]
    y_dna_seqs = [count_cpgs(seq) for seq in test_dna]
    test_dna = test_dna[0]
    return test_dna, y_dna_seqs[0]
