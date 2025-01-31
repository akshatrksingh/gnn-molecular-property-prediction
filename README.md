# GNN for Molecular Properties like HOMO-LUMO Gap

## Project Overview
This project implements a **Graph Neural Network (GNN)** to predict molecular properties using graph-based representations of molecules. It leverages **Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)** to learn molecular embeddings and predict target properties. While the code is designed generically for molecular property prediction, the provided training and evaluation results are specifically for **HOMO-LUMO gap prediction** using the **QM9 dataset**.

## Repository Structure
```
├── models
│   ├── GAT_model.pt               # Trained GAT model
│   ├── GCN_model.pt               # Trained GCN model
├── requirements.txt                # Dependencies
├── src
│   ├── data_processing/            # Data preprocessing scripts
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── dataset_loader.py       # Dataset loading functions
│   ├── metrics/                    # Performance evaluation utilities
│   │   ├── gat.py                   # Metrics for GAT model
│   │   ├── gcn.py                   # Metrics for GCN model
│   ├── models/                     # Model architectures
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── gat.py                   # GAT model implementation
│   │   ├── gcn.py                   # GCN model implementation
│   ├── train.py                    # Training script
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gnn-molecular-property-prediction.git
   cd gnn-molecular-property-prediction
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model
Run the training script:
```bash
python src/train.py
```

The model will train on the **QM9 dataset**, specifically predicting the **HOMO-LUMO gap**, and log evaluation metrics during training.

## Model Performance
After training, the model's performance is evaluated using the following metrics:

### **Graph Convolutional Network (GCN) Results**
| Metric  | Value  |
|---------|--------|
| **MAE** | 0.3425 |
| **MSE** | 0.2043 |
| **R²**  | 0.8051 |

### **Graph Attention Network (GAT) Results**
| Metric  | Value  |
|---------|--------|
| **MAE** | 0.3598 |
| **MSE** | 0.2188 |
| **R²**  | 0.7925 |

These results are for **HOMO-LUMO gap prediction on the QM9 dataset** and indicate a strong correlation between predicted and actual values.
