import torch
import numpy as np
import os
import math
import logging
import argparse
from datetime import datetime
from data_processing.dataset_loader import load_qm9, preprocess_qm9  
from models.gcn import GCN
from models.gat import GAT

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train a GNN model (GCN or GAT).")
parser.add_argument("--model", type=str, choices=["GCN", "GAT"], default="GCN",
                    help="Specify which model to train: GCN or GAT")
args = parser.parse_args()

# Get model name from args
model_name = args.model

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Logger setup - Unique log file per model & date
log_filename = f"logs/training_{model_name}_{current_date}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"Starting training for {model_name} on {current_date}")

def training(loader, model, loss, optimizer):
    """Trains for one epoch"""
    model.train()
    total_loss = 0

    for d in loader:
        optimizer.zero_grad()
        d.x = d.x.float()
        out = model(d)
        l = loss(out, torch.reshape(d.y, (len(d.y), 1)))
        l.backward()
        optimizer.step()
        total_loss += l.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, model

@torch.no_grad()
def validation(loader, model, loss):
    """Evaluates model on validation set"""
    model.eval()
    total_loss = 0

    for d in loader:
        out = model(d)
        l = loss(out, torch.reshape(d.y, (len(d.y), 1)))
        total_loss += l.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

def train_epochs(epochs, model, train_loader, val_loader, path):
    """Trains model for given epochs, logs progress, saves best model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = torch.nn.MSELoss()

    train_target = np.empty((0))
    train_y_target = np.empty((0))
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = math.inf

    # Resume training if checkpoint exists
    if os.path.exists(path):
        logging.info(f"Resuming from checkpoint: {path}")
        model.load_state_dict(torch.load(path))

    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, loss, optimizer)
        v_loss = validation(val_loader, model, loss)

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), path)
            logging.info(f"New best model saved at epoch {epoch}, Val Loss: {v_loss:.4f}")

        for d in train_loader:
            out = model(d)
            if epoch == epochs - 1:
                train_target = np.concatenate((train_target, out.detach().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, d.y.detach().numpy()))

        train_loss[epoch] = epoch_loss
        val_loss[epoch] = v_loss

        logging.info(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Val Loss: {v_loss:.4f}")

    return train_loss, val_loss, train_target, train_y_target

if __name__ == "__main__":
    dataset = load_qm9()
    train_loader, val_loader, test_loader = preprocess_qm9(dataset)  

    epochs = 150

    # Dynamically initialize GCN or GAT
    if model_name == "GCN":
        model = GCN(dim_h=128)
    else:
        model = GAT(dim_h=128)

    model_path = f"models/{model_name}_model.pt"

    train_loss, val_loss, train_target, train_y_target = train_epochs(
        epochs, model, train_loader, val_loader, model_path
    )

    logging.info(f"Training complete. Model saved at: {model_path}")
    print(f"Training complete. Model saved at: {model_path}")