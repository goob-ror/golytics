#!/usr/bin/env python3
"""
Quick Model Optimization Demo
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("output/training", exist_ok=True)
os.makedirs("output/training/models", exist_ok=True)
os.makedirs("output/training/plots", exist_ok=True)

class OptimizedBusinessMLP(nn.Module):
    """Optimized MLP with business constraints and better architecture"""

    def __init__(self, input_size=3, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        output = self.network(x)
        # Apply business constraints
        modal = torch.relu(output[:, 0])  # Modal must be positive
        profit = output[:, 1]  # Profit can be negative
        rugi = torch.relu(output[:, 2])   # Loss must be positive
        return torch.stack([modal, profit, rugi], dim=1)

def load_and_prepare_data():
    """Load and prepare business data"""
    print("📊 Loading business data...")

    # Load data
    df = pd.read_csv("dataset/csv/business_owner_dataset_extended.csv")

    # Extract features and targets
    features = ['pemasukan', 'pengeluaran', 'jam']
    targets = ['modal', 'profit', 'rugi']

    X = df[features].values
    y = df[targets].values

    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def train_optimized_model(X_train, y_train, X_val, y_val, config_name="optimized"):
    """Train model with optimized configuration"""
    print(f"🧠 Training {config_name} model...")

    # Model configuration
    model = OptimizedBusinessMLP(
        input_size=3,
        hidden_sizes=[128, 64, 32],
        output_size=3,
        dropout=0.3
    )

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)

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
    train_r2_scores = []
    val_r2_scores = []
    learning_rates = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25

    print("🚀 Starting training...")

    for epoch in range(200):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            train_predictions.append(output.detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

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

        train_pred = np.vstack(train_predictions)
        train_true = np.vstack(train_targets)
        val_pred = np.vstack(val_predictions)
        val_true = np.vstack(val_targets)

        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)

        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)
        learning_rates.append(current_lr)

        # Update scheduler
        scheduler.step(val_loss)

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"output/training/models/{config_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Progress reporting
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, "
                  f"Val R² {val_r2:.4f}, LR {current_lr:.6f}")

    # Save final model
    torch.save(model.state_dict(), f"output/training/models/{config_name}_final.pth")

    # Create training metrics dictionary
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2_scores': train_r2_scores,
        'val_r2_scores': val_r2_scores,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'best_val_r2': max(val_r2_scores),
        'total_epochs': len(train_losses)
    }

    print(f"✅ Training completed. Best validation loss: {best_val_loss:.6f}, Best R²: {max(val_r2_scores):.4f}")
    return model, metrics

def create_training_visualizations(metrics, config_name):
    """Create comprehensive training visualizations"""
    print(f"📊 Creating training visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs = range(len(metrics['train_losses']))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_losses'], label='Training Loss', linewidth=2)
    ax.plot(epochs, metrics['val_losses'], label='Validation Loss', linewidth=2)
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² scores
    ax = axes[0, 1]
    ax.plot(epochs, metrics['train_r2_scores'], label='Training R²', linewidth=2)
    ax.plot(epochs, metrics['val_r2_scores'], label='Validation R²', linewidth=2)
    ax.set_title('R² Score Progress', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Learning rate schedule
    ax = axes[0, 2]
    ax.plot(epochs, metrics['learning_rates'], linewidth=2, color='orange')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Overfitting analysis
    ax = axes[1, 0]
    gap = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    ax.plot(epochs, gap, linewidth=2, color='red')
    ax.set_title('Overfitting Analysis (Val - Train Loss)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Gap')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 5: Performance summary
    ax = axes[1, 1]
    performance_metrics = ['Best Val Loss', 'Best Val R²', 'Total Epochs']
    performance_values = [metrics['best_val_loss'], metrics['best_val_r2'], metrics['total_epochs']]

    # Normalize for visualization
    normalized_values = [
        1 - metrics['best_val_loss'] / max(metrics['val_losses']),  # Lower is better
        metrics['best_val_r2'],  # Higher is better
        1 - metrics['total_epochs'] / 200  # Fewer epochs is better (efficiency)
    ]

    bars = ax.bar(performance_metrics, normalized_values, alpha=0.7, color=['red', 'green', 'blue'])
    ax.set_title('Performance Summary (Normalized)', fontweight='bold')
    ax.set_ylabel('Normalized Score')
    ax.set_ylim(0, 1)

    # Add actual values as text
    for bar, actual in zip(bars, performance_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{actual:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Training summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"🏆 TRAINING SUMMARY\n\n"
    summary_text += f"Configuration: {config_name}\n\n"
    summary_text += f"📊 Final Metrics:\n"
    summary_text += f"  • Best Val Loss: {metrics['best_val_loss']:.6f}\n"
    summary_text += f"  • Best Val R²: {metrics['best_val_r2']:.4f}\n"
    summary_text += f"  • Total Epochs: {metrics['total_epochs']}\n\n"

    summary_text += f"🎯 Performance Goals:\n"
    if metrics['best_val_r2'] > 0.8:
        summary_text += f"  ✅ Excellent (R² > 0.8)\n"
    elif metrics['best_val_r2'] > 0.7:
        summary_text += f"  ✅ Good (R² > 0.7)\n"
    elif metrics['best_val_r2'] > 0.6:
        summary_text += f"  ⚠️ Acceptable (R² > 0.6)\n"
    else:
        summary_text += f"  ❌ Needs Improvement\n"

    summary_text += f"\n📈 Recommendations:\n"
    if metrics['best_val_r2'] > 0.75:
        summary_text += f"  • Model ready for production\n"
        summary_text += f"  • Consider ensemble methods\n"
    else:
        summary_text += f"  • Try different architectures\n"
        summary_text += f"  • Collect more data\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'output/training/plots/{config_name}_training_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training dashboard saved: {config_name}_training_dashboard.png")

def test_model_performance(model, X_test, y_test, scaler_x, scaler_y, config_name):
    """Test model performance and create test visualizations"""
    print(f"🧪 Testing {config_name} model performance...")

    # Scale test data and make predictions
    X_test_scaled = scaler_x.transform(X_test)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        predictions_scaled = model(X_tensor).numpy()

    # Inverse transform predictions
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Per-target metrics
    target_names = ['Modal', 'Profit', 'Rugi']
    per_target_metrics = {}

    for i, target in enumerate(target_names):
        target_mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        target_r2 = r2_score(y_test[:, i], predictions[:, i])
        per_target_metrics[target] = {'mae': target_mae, 'r2': target_r2}

    # Business constraints check
    modal_positive = np.all(predictions[:, 0] >= 0)
    rugi_positive = np.all(predictions[:, 2] >= 0)

    print(f"  📈 Overall R²: {r2:.4f}")
    print(f"  📉 Overall MAE: {mae:.2f}")
    print(f"  ✅ Modal positive: {modal_positive}")
    print(f"  ✅ Rugi positive: {rugi_positive}")

    # Create test visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot predictions vs actual for each target
    for i, target in enumerate(target_names):
        if i < 3:  # We only have 3 targets but 4 subplots
            row, col = i // 2, i % 2
            ax = axes[row, col]

            ax.scatter(y_test[:, i], predictions[:, i], alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(y_test[:, i].min(), predictions[:, i].min())
            max_val = max(y_test[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            ax.set_xlabel(f'Actual {target}')
            ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'{target} Predictions\nR² = {per_target_metrics[target]["r2"]:.3f}')
            ax.grid(True, alpha=0.3)

    # Summary plot
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"🧪 TEST RESULTS\n\n"
    summary_text += f"Overall Performance:\n"
    summary_text += f"  • R² Score: {r2:.4f}\n"
    summary_text += f"  • MAE: {mae:.2f}\n\n"

    summary_text += f"Per-Target R² Scores:\n"
    for target, metrics in per_target_metrics.items():
        summary_text += f"  • {target}: {metrics['r2']:.3f}\n"

    summary_text += f"\nBusiness Constraints:\n"
    summary_text += f"  • Modal ≥ 0: {'✅' if modal_positive else '❌'}\n"
    summary_text += f"  • Rugi ≥ 0: {'✅' if rugi_positive else '❌'}\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'output/training/plots/{config_name}_test_results.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Test results saved: {config_name}_test_results.png")

    return {
        'overall_r2': r2,
        'overall_mae': mae,
        'per_target_metrics': per_target_metrics,
        'constraints_passed': modal_positive and rugi_positive
    }

def main():
    """Main execution function"""
    print("🚀 QUICK MODEL OPTIMIZATION DEMO")
    print("=" * 60)

    # Load and prepare data
    X, y = load_and_prepare_data()

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"📊 Data splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_x.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    # Save scalers
    joblib.dump(scaler_x, "output/training/models/scaler_x_optimized.pkl")
    joblib.dump(scaler_y, "output/training/models/scaler_y_optimized.pkl")

    # Train optimized model
    model, metrics = train_optimized_model(
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
        config_name="optimized"
    )

    # Create training visualizations
    create_training_visualizations(metrics, "optimized")

    # Test model performance
    test_results = test_model_performance(
        model, X_test, y_test, scaler_x, scaler_y, "optimized"
    )

    # Save comprehensive results
    final_results = {
        'training_metrics': metrics,
        'test_results': test_results,
        'model_config': {
            'architecture': 'OptimizedBusinessMLP',
            'hidden_sizes': [128, 64, 32],
            'dropout': 0.3,
            'optimizer': 'AdamW',
            'learning_rate': 0.003,
            'weight_decay': 1e-4,
            'scheduler': 'ReduceLROnPlateau'
        }
    }

    with open('output/training/optimized_model_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n🎉 OPTIMIZATION DEMO COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best Validation R²: {metrics['best_val_r2']:.4f}")
    print(f"🧪 Test R²: {test_results['overall_r2']:.4f}")
    print(f"✅ Business Constraints: {'Passed' if test_results['constraints_passed'] else 'Failed'}")
    print(f"📁 Results saved in: output/training/")
    print(f"📊 Visualizations: output/training/plots/")
    print(f"🤖 Models: output/training/models/")

    return final_results

if __name__ == "__main__":
    results = main()
    print("\n✅ Demo completed successfully!")
