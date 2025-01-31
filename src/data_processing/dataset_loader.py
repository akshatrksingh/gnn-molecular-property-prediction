import torch
import pandas as pd
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def load_qm9(root="data"):
    """Loads the QM9 dataset."""
    dataset = QM9(root=root)
    return dataset

def preprocess_qm9(qm9):
    """
    Preprocesses the QM9 dataset:
    - Selects a single regression target (index 4)
    - Normalizes target values
    - Splits into train, validation, and test sets
    - Returns DataLoaders
    """
    # Select one regression target
    y_target = pd.DataFrame(qm9.data.y.numpy())
    qm9.data.y = torch.Tensor(y_target[4])

    # Shuffle dataset
    qm9 = qm9.shuffle()

    # Data split
    data_size = len(qm9)
    train_index = int(data_size * 0.8)
    test_index = train_index + int(data_size * 0.1)
    val_index = test_index + int(data_size * 0.1)

    # Normalizing the target variable
    data_mean = qm9.data.y[0:train_index].mean()
    data_std = qm9.data.y[0:train_index].std()
    qm9.data.y = (qm9.data.y - data_mean) / data_std

    # Create DataLoaders
    train_loader = DataLoader(qm9[0:train_index], batch_size=64, shuffle=True)
    test_loader = DataLoader(qm9[train_index:test_index], batch_size=64, shuffle=True)
    val_loader = DataLoader(qm9[test_index:val_index], batch_size=64, shuffle=True)

    return train_loader, test_loader, val_loader

if __name__ == "__main__":
    dataset = load_qm9()
    train_loader, test_loader, val_loader = preprocess_qm9(dataset)
    
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
