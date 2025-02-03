import torch
import random
from src.utils import prepare_data, dna2int, set_seed, create_test_data
from src.dataset import CPGDatasetPackPadding
from src.model import CpGCounterAdvancedPackPadding
from src.trainer import (
    training_loop_pack_padded,
    validation_loop_pack_padded,
    train_model_pack_padded,
)
from src.utils import (
    predict_cpgs_from_dna_pack_padded,
    save_checkpoint,
    load_checkpoint,
)
from src.tuner import tune_hyperparameters

##### -------------------------------------------------------- Hyperparameters --------------------------------------------------------
vocab_size = len(dna2int)
batch_size = 16
embedding_dim = 64
hidden_size = 256
num_layers = 2
dropout = 0.2
learning_rate = 0.001
num_epochs = 100
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_patience = 10
save_model_path = "best_cpg_model_advanced_packpad.pth"
optuna_model_path = "best_cpg_model_optuna.pth"

##### --------------------- Prepare Data ---------------------
if __name__ == "__main__":
    set_seed(42)

    # Generate training and test data
    print("Generating training and test data...")
    train_x, train_y = prepare_data("variable", 2048)
    test_x, test_y = prepare_data("variable", 512)

    # Print dataset statistics
    print(
        f"Train Sequence Lengths - Min: {min(map(len, train_x))}, Max: {max(map(len, train_x))}"
    )
    print(
        f"Test Sequence Lengths - Min: {min(map(len, test_x))}, Max: {max(map(len, test_x))}"
    )
    print(f"Total Training Samples: {len(train_x)}, Testing Samples: {len(test_x)}")

    # Create Dataset and Dataloaders
    print("Creating dataset and dataloaders...")
    train_dataset = CPGDatasetPackPadding(train_x, train_y)
    val_dataset = CPGDatasetPackPadding(test_x, test_y)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CPGDatasetPackPadding.collate_fn,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CPGDatasetPackPadding.collate_fn,
    )

    ##### ------------------- Model Training and Evaluation -------------------
    print("Initializing model for training...")
    model = CpGCounterAdvancedPackPadding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    print("Starting model training...")
    trained_model = train_model_pack_padded(
        model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=num_epochs,
        patience=stop_patience,
        save_path=save_model_path,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create a test dataset
    print("Generating a test sequence for prediction...")
    test_dna, count_cpg = create_test_data()
    print(f"Test DNA: {test_dna}\nActual CpG Count: {count_cpg}")

    # Test prediction from trained model
    predicted_cpgs = predict_cpgs_from_dna_pack_padded(
        save_model_path,
        test_dna,
        dna2int,
        embedding_dim,
        hidden_size,
        num_layers,
        dropout,
        device,
        model_class=CpGCounterAdvancedPackPadding,
    )

    print("✅ Model Inference Successful!")
    print(f"🔹 DNA: {test_dna}\n🔹 Predicted CpG Count: {predicted_cpgs}")

    ##### ------------------- Hyperparameter Tuning -------------------
    print("Starting Hyperparameter Tuning using Optuna...")

    best_hyperparams, trained_model = tune_hyperparameters(
        vocab_size,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs,
        stop_patience,
        n_trials=7,
        save_best_model_path=optuna_model_path,
        study_name="cpg_optuna",
    )

    # Test prediction from the Optuna-tuned model
    predicted_cpgs = predict_cpgs_from_dna_pack_padded(
        model_path=optuna_model_path,
        dna_sequence=test_dna,
        dna2int=dna2int,
        embedding_dim=best_hyperparams["embedding_dim"],
        hidden_size=best_hyperparams["hidden_size"],
        num_layers=best_hyperparams["num_layers"],
        dropout=best_hyperparams["dropout"],
        device=device,
        model_class=CpGCounterAdvancedPackPadding,
    )

    print("✅ Optuna-Tuned Model Inference Successful!")
    print(
        f"🔹 DNA: {test_dna}\n🔹 Predicted CpG Count (Optuna Model): {predicted_cpgs}"
    )
