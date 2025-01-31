import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os

# Logger setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/evaluation_gat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@torch.no_grad()
def evaluate_model(loader, model, device="cpu"):
    """
    Evaluates the trained model on the test set.

    Args:
        loader (DataLoader): Test DataLoader.
        model (nn.Module): Trained GNN model.
        device (str): Device to run the evaluation on (default: "cpu").

    Returns:
        dict: Dictionary of evaluation metrics (MAE, MSE, R² Score).
    """
    model.to(device)
    model.eval()
    
    true_vals, pred_vals = [], []

    for d in loader:
        d = d.to(device)
        d.x = d.x.float()
        output = model(d)
        
        true_vals.extend(d.y.cpu().numpy())
        pred_vals.extend(output.cpu().detach().numpy().flatten())

    # Compute metrics
    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)

    # Log results
    logging.info(f"Evaluation Results: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}")

    return {"MAE": mae, "MSE": mse, "R²": r2}

if __name__ == "__main__":
    from data_processing.dataset_loader import load_qm9, preprocess_qm9  
    from models.gat import GAT  

    dataset = load_qm9()
    _, _, test_loader = preprocess_qm9(dataset)  

    model = GAT(dim_h=128)
    model.load_state_dict(torch.load("models/GAT_model.pt"))  # Load trained model

    metrics = evaluate_model(test_loader, model)
    print(metrics)