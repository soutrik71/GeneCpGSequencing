import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict

# Import model and inference function
from src.utils import dna2int, predict_cpgs_from_dna_pack_padded
from src.model import CpGCounterAdvancedPackPadding

# Initialize FastAPI app with metadata
app = FastAPI(
    title="CpG Predictor API",
    description="A FastAPI-based service for predicting CpG counts from DNA sequences using an LSTM model.",
    version="1.0.0",
    contact={
        "name": "soutrik",
        "email": "soutrik1991@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_cpg_model_advanced_packpad.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters (Ensure they match training)
embedding_dim = 64
hidden_size = 256
num_layers = 2
dropout = 0.2


# Define the request schema
class DNASequenceRequest(BaseModel):
    dna_sequence: str = Field(
        description="A valid DNA sequence containing 'A', 'C', 'G', 'T', and 'N'."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dna_sequence": "CCGTTTAANAATGATTAAACNGCCGTGCATACTGCANGGGNTATNNATTGGNNNCGAGTTANCNA"
            }
        }


# Define the response schema
class DNASequenceResponse(BaseModel):
    dna_sequence: str
    predicted_cpg_count: float

    class Config:
        json_schema_extra = {
            "example": {
                "dna_sequence": "CCGTTTAANAATGATTAAACNGCCGTGCATACTGCANGGGNTATNNATTGGNNNCGAGTTANCNA",
                "predicted_cpg_count": 4.7,
            }
        }


# Load the model at startup
@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join("models", MODEL_PATH)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model checkpoint not found at {model_path}")

    # Initialize Model
    model = CpGCounterAdvancedPackPadding(
        vocab_size=len(dna2int),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Load Model Weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully from {MODEL_PATH}")


# Health Check Endpoint
@app.get("/health", tags=["Utility"])
def health_check():
    """
    Check if the model is loaded and available.
    """
    return {"status": "Model is loaded and ready!"}


# Prediction Endpoint
@app.post("/predict", response_model=DNASequenceResponse, tags=["Prediction"])
def predict_cpg(request: DNASequenceRequest):
    """
    Predict CpG count from a DNA sequence.
    """
    try:
        predicted_cpgs = predict_cpgs_from_dna_pack_padded(
            model_path=MODEL_PATH,
            dna_sequence=request.dna_sequence,
            dna2int=dna2int,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=DEVICE,
            model_class=CpGCounterAdvancedPackPadding,
        )
        return DNASequenceResponse(
            dna_sequence=request.dna_sequence, predicted_cpg_count=predicted_cpgs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
