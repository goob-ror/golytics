#!/usr/bin/env python3
"""
Simple Training Demo with Optimized Configuration
Demonstrates improved training with better configurations and visualizations
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("output/training", exist_ok=True)
os.makedirs("output/training/plots", exist_ok=True)

class ImprovedBusinessMLP(nn.Module):
    """Improved MLP with business constraints"""

    def __init__(self, input_size=3, hidden_size=64, output_size=3, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        output = self.network(x)
        # Apply business constraints
        modal = torch.relu(output[:, 0])  # Modal must be positive
        profit = output[:, 1]  # Profit can be negative
        rugi = torch.relu(output[:, 2])   # Loss must be positive
        return torch.stack([modal, profit, rugi], dim=1)

def load_data():
    """Load the processed data"""
    print("📊 Loading processed data...")

    X_scaled = pd.read_csv("ensemble/data/X_scaled.csv").values
    y_scaled = pd.read_csv("ensemble/data/y_scaled.csv").values
    scaler_x = joblib.load("ensemble/data/scaler_x.pkl")
    scaler_y = joblib.load("ensemble/data/scaler_y.pkl")

    print(f"✅ Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y_scaled, scaler_x, scaler_y

def train_improved_model(X_train, y_train, X_val, y_val):
    """Train model with improved configuration"""
    print("🧠 Training improved model...")

    # Model setup
    model = ImprovedBusinessMLP(input_size=3, hidden_size=128, output_size=3, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Training tracking
    train_losses = []
    val_losses = []
    val_r2_scores = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    print("🚀 Starting training...")

    for epoch in range(100):  # Reduced epochs for demo
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

                val_predictions.append(output.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        val_pred = np.vstack(val_predictions)
        val_true = np.vstack(val_targets)
        val_r2 = r2_score(val_true, val_pred)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "output/training/improved_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Progress reporting
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, Val R² {val_r2:.4f}")

    print(f"✅ Training completed. Best validation loss: {best_val_loss:.6f}, Best R²: {max(val_r2_scores):.4f}")

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_scores': val_r2_scores,
        'best_val_loss': best_val_loss,
        'best_val_r2': max(val_r2_scores)
    }

def create_training_plots(metrics):
    """Create training visualization plots"""
    print("📊 Creating training visualizations...")

    epochs = range(len(metrics['train_losses']))

    # Create training dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_losses'], label='Training Loss', linewidth=2, color='blue')
    ax.plot(epochs, metrics['val_losses'], label='Validation Loss', linewidth=2, color='red')
    ax.set_title('Training & Validation Loss', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² score
    ax = axes[0, 1]
    ax.plot(epochs, metrics['val_r2_scores'], linewidth=2, color='green')
    ax.set_title('Validation R² Score Progress', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.grid(True, alpha=0.3)

    # Plot 3: Performance summary
    ax = axes[1, 0]
    performance_metrics = ['Best Val Loss', 'Best Val R²', 'Total Epochs']
    performance_values = [
        metrics['best_val_loss'],
        metrics['best_val_r2'],
        len(metrics['train_losses'])
    ]

    bars = ax.bar(performance_metrics, performance_values, alpha=0.7, color=['red', 'green', 'blue'])
    ax.set_title('Performance Summary', fontweight='bold', fontsize=14)
    ax.set_ylabel('Value')

    # Add value labels
    for bar, value in zip(bars, performance_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(performance_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Training summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"🏆 TRAINING SUMMARY\n\n"
    summary_text += f"📊 Final Metrics:\n"
    summary_text += f"  • Best Val Loss: {metrics['best_val_loss']:.6f}\n"
    summary_text += f"  • Best Val R²: {metrics['best_val_r2']:.4f}\n"
    summary_text += f"  • Total Epochs: {len(metrics['train_losses'])}\n\n"

    summary_text += f"🎯 Performance Assessment:\n"
    if metrics['best_val_r2'] > 0.8:
        summary_text += f"  ✅ Excellent Performance\n"
    elif metrics['best_val_r2'] > 0.7:
        summary_text += f"  ✅ Good Performance\n"
    elif metrics['best_val_r2'] > 0.6:
        summary_text += f"  ⚠️ Acceptable Performance\n"
    else:
        summary_text += f"  ❌ Needs Improvement\n"

    summary_text += f"\n📈 Model Features:\n"
    summary_text += f"  • Business Constraints Applied\n"
    summary_text += f"  • Xavier Weight Initialization\n"
    summary_text += f"  • Dropout Regularization\n"
    summary_text += f"  • Early Stopping\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('output/training/plots/improved_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Training visualization saved: improved_training_results.png")

def test_model(model, X_test, y_test, scaler_x, scaler_y):
    """Test the trained model"""
    print("🧪 Testing model performance...")

    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions_scaled = model(X_tensor).numpy()

    # Inverse transform to original scale
    predictions = scaler_y.inverse_transform(predictions_scaled)
    actual = scaler_y.inverse_transform(y_test)

    # Calculate metrics
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    # Business constraints check
    modal_positive = np.all(predictions[:, 0] >= 0)
    rugi_positive = np.all(predictions[:, 2] >= 0)

    print(f"  📈 Test R²: {r2:.4f}")
    print(f"  📉 Test MAE: {mae:.2f}")
    print(f"  ✅ Modal positive: {modal_positive}")
    print(f"  ✅ Rugi positive: {rugi_positive}")

    # Create test visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Modal', 'Profit', 'Rugi']

    for i, target in enumerate(target_names):
        ax = axes[i]

        ax.scatter(actual[:, i], predictions[:, i], alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(actual[:, i].min(), predictions[:, i].min())
        max_val = max(actual[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        target_r2 = r2_score(actual[:, i], predictions[:, i])
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{target} Predictions\nR² = {target_r2:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/training/plots/test_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Test visualization saved: test_results.png")

    return {
        'r2': float(r2),
        'mae': float(mae),
        'constraints_passed': bool(modal_positive and rugi_positive)
    }

def main():
    """Main execution function"""
    print("🚀 SIMPLE TRAINING OPTIMIZATION DEMO")
    print("=" * 60)

    # Load data
    X_scaled, y_scaled, scaler_x, scaler_y = load_data()

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"📊 Data splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

    # Train model
    model, metrics = train_improved_model(X_train, y_train, X_val, y_val)

    # Create training visualizations
    create_training_plots(metrics)

    # Test model
    test_results = test_model(model, X_test, y_test, scaler_x, scaler_y)

    # Save results (convert numpy types to Python types for JSON serialization)
    final_results = {
        'training_metrics': {
            'train_losses': [float(x) for x in metrics['train_losses']],
            'val_losses': [float(x) for x in metrics['val_losses']],
            'val_r2_scores': [float(x) for x in metrics['val_r2_scores']],
            'best_val_loss': float(metrics['best_val_loss']),
            'best_val_r2': float(metrics['best_val_r2'])
        },
        'test_results': {
            'r2': float(test_results['r2']),
            'mae': float(test_results['mae']),
            'constraints_passed': bool(test_results['constraints_passed'])
        },
        'model_architecture': 'ImprovedBusinessMLP',
        'improvements': [
            'Business constraints applied',
            'Xavier weight initialization',
            'Dropout regularization',
            'Early stopping',
            'Larger hidden layer (128 units)',
            'Weight decay regularization'
        ]
    }

    with open('output/training/simple_optimization_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n🎉 OPTIMIZATION DEMO COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best Validation R²: {metrics['best_val_r2']:.4f}")
    print(f"🧪 Test R²: {test_results['r2']:.4f}")
    print(f"✅ Business Constraints: {'Passed' if test_results['constraints_passed'] else 'Failed'}")
    print(f"📁 Results saved in: output/training/")
    print(f"📊 Visualizations: output/training/plots/")

    return final_results

if __name__ == "__main__":
    results = main()
    print("\n✅ Demo completed successfully!")
