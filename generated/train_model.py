"""
Machine Learning Modell für Swaption-Preis Vorhersagen
Trainiert auf Basis der train.xlsx Daten
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SwaptionDataset(Dataset):
    """PyTorch Dataset für Swaptionc-Daten"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Features (n_samples, n_features) - Swaption-Preise eines Tages
            y: Targets (n_samples, n_features) - Swaption-Preise des nächsten Tages
        """
        self.X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SwaptionPredictor(nn.Module):
    """Klassisches Neural Network für Swaption-Preis Vorhersagen"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.2):
        super(SwaptionPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, input_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class QuantumSwaptionPredictor(nn.Module):
    """Quantum-basiertes Modell mit MerLin QuantumLayer"""
    
    def __init__(self, input_size, n_modes=10, n_photons=5, n_params=100, dtype=torch.float32):
        super(QuantumSwaptionPredictor, self).__init__()
        
        try:
            from merlin import QuantumLayer, LexGrouping
            from merlin.builder import CircuitBuilder
            
            # Quantum Layer
            self.quantum_layer = QuantumLayer.simple(
                input_size=input_size,
                n_params=n_params,
                dtype=dtype,
            )
            
            # Grouping to reduce quantum output
            reduced_size = min(64, self.quantum_layer.output_size)
            self.grouping = LexGrouping(self.quantum_layer.output_size, reduced_size)
            
            # Classical head for final prediction
            self.classical_head = nn.Sequential(
                nn.Linear(reduced_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, input_size)
            )
            
        except ImportError:
            print("Warning: MerLin nicht verfügbar, verwende klassisches Modell")
            raise ImportError("MerLin ist erforderlich für QuantumSwaptionPredictor")
    
    def forward(self, x):
        quantum_out = self.quantum_layer(x)
        grouped = self.grouping(quantum_out)
        return self.classical_head(grouped)


def load_and_prepare_data(data_path):
    """Lädt und bereitet die Daten vor"""
    print(f"Lade Daten von {data_path}...")
    df = pd.read_excel(data_path)
    
    # Date-Spalte konvertieren
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Feature-Spalten (alle außer Date)
    feature_cols = [c for c in df.columns if c != 'Date']
    
    print(f"Anzahl Zeilen: {len(df)}")
    print(f"Anzahl Features: {len(feature_cols)}")
    print(f"Date Range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    
    # Features und Targets vorbereiten (nächster Tag als Target)
    X = df[feature_cols].copy()
    y = df[feature_cols].shift(-1)
    
    # Letzte Zeile entfernen (hat kein Target)
    mask = ~y.isnull().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"Anzahl Samples nach Bereinigung: {len(X)}")
    
    return X, y, feature_cols


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Trainiert das Modell"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarte Training auf {device}...")
    print(f"Anzahl trainierbare Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Bestes Modell speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Bestes Modell laden
    model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses, model


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluiert das Modell auf Test-Daten"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # Metriken berechnen
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Relative Fehler
    relative_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))
    
    print(f"\n=== Evaluation Ergebnisse ===")
    print(f"MSE Loss: {avg_loss:.6f}")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"Relative Error: {relative_error:.4%}")
    
    return {
        'mse': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'relative_error': relative_error,
        'predictions': predictions,
        'targets': targets
    }


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plottet die Training-Kurven"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training und Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining-Kurven gespeichert: {save_path}")
    plt.close()


def main():
    """Hauptfunktion"""
    # Pfade
    data_path = '../provided/train.xlsx'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}")
    
    # Daten laden
    X, y, feature_cols = load_and_prepare_data(data_path)
    
    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # shuffle=False für Zeitreihen
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalisierung
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Datasets und DataLoaders
    train_dataset = SwaptionDataset(X_train_scaled, y_train_scaled)
    val_dataset = SwaptionDataset(X_val_scaled, y_val_scaled)
    test_dataset = SwaptionDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Modell erstellen
    input_size = len(feature_cols)
    
    # Versuche Quantum-Modell, sonst klassisches
    # Hinweis: Quantum-Modell funktioniert nur wenn input_size <= n_modes
    # Für 224 Features brauchen wir mindestens 224 Modi, was sehr rechenintensiv ist
    use_quantum = False
    try:
        # Quantum-Modell nur für kleinere Feature-Sets sinnvoll
        # Für große Feature-Sets (wie hier mit 224) verwenden wir klassisches Modell
        if input_size <= 20:  # Nur für kleine Feature-Sets
            model = QuantumSwaptionPredictor(input_size=input_size)
            use_quantum = True
            print("\nVerwende Quantum-basiertes Modell (MerLin)")
        else:
            raise ValueError("Zu viele Features für Quantum-Modell")
    except (ImportError, ValueError) as e:
        print(f"\nVerwende klassisches Neural Network (Grund: {e})")
        model = SwaptionPredictor(input_size=input_size, hidden_sizes=[256, 128, 64])
    
    # Training
    train_losses, val_losses, trained_model = train_model(
        model, train_loader, val_loader, epochs=100, lr=0.001, device=device
    )
    
    # Evaluation
    results = evaluate_model(trained_model, test_loader, device=device)
    
    # Plots
    plot_training_curves(train_losses, val_losses)
    
    # Modell speichern
    model_path = 'swaption_model.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'model_type': 'quantum' if use_quantum else 'classical'
    }, model_path)
    print(f"\nModell gespeichert: {model_path}")
    
    print("\n=== Training abgeschlossen ===")


if __name__ == '__main__':
    main()

