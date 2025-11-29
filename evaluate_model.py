"""
Evaluations-Skript für das trainierte Swaption-Modell
Analysiert die Modell-Performance mit detaillierten Metriken und Visualisierungen
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train_model import SwaptionDataset, SwaptionPredictor, QuantumSwaptionPredictor, load_and_prepare_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_trained_model(model_path='swaption_model.pth', device='cpu'):
    """Lädt das trainierte Modell"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Modell-Typ bestimmen
    model_type = checkpoint.get('model_type', 'classical')
    input_size = len(checkpoint['feature_cols'])
    
    # Modell erstellen
    if model_type == 'quantum':
        model = QuantumSwaptionPredictor(input_size=input_size)
    else:
        model = SwaptionPredictor(input_size=input_size, hidden_sizes=[256, 128, 64])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, checkpoint['scaler_X'], checkpoint['scaler_y'], checkpoint['feature_cols']


def evaluate_detailed(model, test_loader, scaler_y, device='cpu'):
    """Detaillierte Evaluation mit verschiedenen Metriken"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_inputs.append(X_batch.cpu().numpy())
    
    # Zusammenführen
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)
    
    # Zurücktransformieren (von normalisierten zu originalen Werten)
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Globale Metriken
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, predictions_original)
    
    # Relative Metriken
    relative_error = np.abs(predictions_original - targets_original) / (np.abs(targets_original) + 1e-8)
    mean_relative_error = np.mean(relative_error)
    median_relative_error = np.median(relative_error)
    
    # Per-Feature Metriken
    feature_mae = np.mean(np.abs(predictions_original - targets_original), axis=0)
    feature_rmse = np.sqrt(np.mean((predictions_original - targets_original) ** 2, axis=0))
    
    print("=" * 60)
    print("DETAILLIERTE EVALUATION")
    print("=" * 60)
    print(f"\n📊 Globale Metriken:")
    print(f"  MSE (Mean Squared Error):     {mse:.6f}")
    print(f"  MAE (Mean Absolute Error):    {mae:.6f}")
    print(f"  RMSE (Root Mean Squared):     {rmse:.6f}")
    print(f"  R² Score:                     {r2:.4f}")
    print(f"  Mean Relative Error:           {mean_relative_error:.2%}")
    print(f"  Median Relative Error:        {median_relative_error:.2%}")
    
    print(f"\n📈 Per-Feature Statistiken:")
    print(f"  Beste Vorhersage (niedrigster MAE): Feature {np.argmin(feature_mae)} (MAE: {np.min(feature_mae):.6f})")
    print(f"  Schlechteste Vorhersage (höchster MAE): Feature {np.argmax(feature_mae)} (MAE: {np.max(feature_mae):.6f})")
    print(f"  Durchschnittlicher Feature-MAE: {np.mean(feature_mae):.6f}")
    print(f"  Durchschnittlicher Feature-RMSE: {np.mean(feature_rmse):.6f}")
    
    return {
        'predictions': predictions_original,
        'targets': targets_original,
        'inputs': inputs,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'feature_mae': feature_mae,
        'feature_rmse': feature_rmse
    }


def plot_predictions_vs_targets(results, save_dir='evaluation_plots'):
    """Plottet Vorhersagen vs. tatsächliche Werte"""
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    
    # 1. Scatter Plot: Vorhersagen vs. Targets
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=1)
    
    # Perfekte Vorhersage-Linie
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfekte Vorhersage')
    
    plt.xlabel('Tatsächliche Werte', fontsize=12)
    plt.ylabel('Vorhergesagte Werte', fontsize=12)
    plt.title('Vorhersagen vs. Tatsächliche Werte', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R² in Plot einfügen
    r2 = results['r2']
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predictions_vs_targets.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot gespeichert: {save_dir}/predictions_vs_targets.png")
    plt.close()
    
    # 2. Residual Plot
    residuals = predictions.flatten() - targets.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(targets.flatten(), residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Tatsächliche Werte', fontsize=12)
    plt.ylabel('Residuen (Vorhersage - Tatsächlich)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/residuals.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {save_dir}/residuals.png")
    plt.close()
    
    # 3. Fehler-Verteilung
    errors = np.abs(predictions - targets).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Absoluter Fehler', fontsize=12)
    plt.ylabel('Häufigkeit', fontsize=12)
    plt.title('Verteilung der Absoluten Fehler', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.4f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {save_dir}/error_distribution.png")
    plt.close()


def plot_time_series_predictions(results, test_indices, feature_cols, save_dir='evaluation_plots'):
    """Plottet Zeitreihen-Vorhersagen für ausgewählte Features"""
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    
    # Wähle einige repräsentative Features (z.B. erste, mittlere, letzte)
    n_features = len(feature_cols)
    selected_features = [0, n_features // 4, n_features // 2, 3 * n_features // 4, n_features - 1]
    
    fig, axes = plt.subplots(len(selected_features), 1, figsize=(14, 3 * len(selected_features)))
    if len(selected_features) == 1:
        axes = [axes]
    
    for ax, feat_idx in zip(axes, selected_features):
        ax.plot(targets[:, feat_idx], label='Tatsächlich', alpha=0.7, linewidth=2)
        ax.plot(predictions[:, feat_idx], label='Vorhersage', alpha=0.7, linewidth=2, linestyle='--')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Wert', fontsize=10)
        ax.set_title(f'Feature {feat_idx}: {feature_cols[feat_idx][:50]}...', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_series_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {save_dir}/time_series_predictions.png")
    plt.close()


def plot_feature_performance(results, feature_cols, save_dir='evaluation_plots', top_n=20):
    """Plottet Performance pro Feature"""
    os.makedirs(save_dir, exist_ok=True)
    
    feature_mae = results['feature_mae']
    
    # Top N beste und schlechteste Features
    sorted_indices = np.argsort(feature_mae)
    worst_indices = sorted_indices[-top_n:]
    best_indices = sorted_indices[:top_n]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Beste Features
    ax1.barh(range(top_n), feature_mae[best_indices])
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f'Feat {i}' for i in best_indices], fontsize=8)
    ax1.set_xlabel('MAE', fontsize=10)
    ax1.set_title(f'Top {top_n} beste Features (niedrigster MAE)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Schlechteste Features
    ax2.barh(range(top_n), feature_mae[worst_indices])
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f'Feat {i}' for i in worst_indices], fontsize=8)
    ax2.set_xlabel('MAE', fontsize=10)
    ax2.set_title(f'Top {top_n} schlechteste Features (höchster MAE)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_performance.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {save_dir}/feature_performance.png")
    plt.close()


def main():
    """Hauptfunktion"""
    print("🔍 Lade Modell und Daten...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Device: {device}\n")
    
    # Modell laden
    model, scaler_X, scaler_y, feature_cols = load_trained_model('swaption_model.pth', device)
    print(f"✅ Modell geladen: {len(feature_cols)} Features\n")
    
    # Daten laden
    X, y, _ = load_and_prepare_data('provided/train.xlsx')
    
    # Train/Val/Test Split (gleicher wie beim Training)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Normalisierung
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Dataset und DataLoader
    test_dataset = SwaptionDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test-Set Größe: {len(test_dataset)} Samples\n")
    
    # Evaluation
    results = evaluate_detailed(model, test_loader, scaler_y, device)
    
    # Visualisierungen
    print("\n📊 Erstelle Visualisierungen...")
    plot_predictions_vs_targets(results)
    plot_time_series_predictions(results, range(len(X_test)), feature_cols)
    plot_feature_performance(results, feature_cols)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation abgeschlossen!")
    print("=" * 60)
    print(f"\nAlle Plots gespeichert im Ordner: evaluation_plots/")


if __name__ == '__main__':
    main()

