#!/usr/bin/env python3
"""
Optimized Training Demo using existing processed data
Demonstrates improved training with better configurations and detailed visualizations
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("output/training", exist_ok=True)
os.makedirs("output/training/models", exist_ok=True)
os.makedirs("output/training/plots", exist_ok=True)
os.makedirs("output/training/logs", exist_ok=True)

class OptimizedBusinessMLP(nn.Module):
    """Optimized MLP with business constraints and advanced architecture"""

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

        # Initialize weights with Xavier initialization
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

def load_processed_data():
    """Load the existing processed data"""
    print("📊 Loading processed data...")

    # Load scaled data
    X_scaled = pd.read_csv("ensemble/data/X_scaled.csv").values
    y_scaled = pd.read_csv("ensemble/data/y_scaled.csv").values

    # Load scalers
    scaler_x = joblib.load("ensemble/data/scaler_x.pkl")
    scaler_y = joblib.load("ensemble/data/scaler_y.pkl")

    print(f"✅ Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y_scaled, scaler_x, scaler_y

def train_optimized_model(X_train, y_train, X_val, y_val, config):
    """Train model with optimized configuration"""
    print(f"🧠 Training {config['name']} model...")

    # Create model
    model = OptimizedBusinessMLP(
        input_size=3,
        hidden_sizes=config['hidden_sizes'],
        output_size=3,
        dropout=config['dropout']
    )

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🔧 Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Setup scheduler
    if config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    else:  # cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Training tracking
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_r2_scores': [],
        'val_r2_scores': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': [],
        'epochs': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    print("🚀 Starting training...")

    for epoch in range(config['epochs']):
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

            # Gradient clipping for stability
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
        train_mae = mean_absolute_error(train_true, train_pred)
        val_mae = mean_absolute_error(val_true, val_pred)

        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        metrics['epochs'].append(epoch)
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_r2_scores'].append(train_r2)
        metrics['val_r2_scores'].append(val_r2)
        metrics['train_mae'].append(train_mae)
        metrics['val_mae'].append(val_mae)
        metrics['learning_rates'].append(current_lr)

        # Update scheduler
        if config['scheduler'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"output/training/models/{config['name']}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Progress reporting
        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, "
                  f"Val R² {val_r2:.4f}, Val MAE {val_mae:.4f}, LR {current_lr:.6f}")

    # Save final model and metrics
    torch.save(model.state_dict(), f"output/training/models/{config['name']}_final.pth")

    metrics['best_val_loss'] = best_val_loss
    metrics['best_val_r2'] = max(metrics['val_r2_scores'])
    metrics['best_val_mae'] = min(metrics['val_mae'])
    metrics['total_epochs'] = len(metrics['epochs'])

    # Save metrics to JSON
    with open(f"output/training/logs/{config['name']}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Training completed. Best validation loss: {best_val_loss:.6f}, Best R²: {metrics['best_val_r2']:.4f}")
    return model, metrics

def create_comprehensive_visualizations(all_results):
    """Create comprehensive training visualizations comparing all strategies"""
    print("📊 Creating comprehensive training visualizations...")

    # Create main comparison dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Plot 1: Loss curves comparison
    ax = axes[0, 0]
    for config_name, (model, metrics) in all_results.items():
        epochs = metrics['epochs']
        ax.plot(epochs, metrics['train_losses'], label=f'{config_name} Train', alpha=0.7)
        ax.plot(epochs, metrics['val_losses'], label=f'{config_name} Val', linestyle='--', alpha=0.7)
    ax.set_title('Training & Validation Loss Comparison', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² scores comparison
    ax = axes[0, 1]
    for config_name, (model, metrics) in all_results.items():
        epochs = metrics['epochs']
        ax.plot(epochs, metrics['val_r2_scores'], label=config_name, linewidth=2)
    ax.set_title('Validation R² Score Progress', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: MAE comparison
    ax = axes[0, 2]
    for config_name, (model, metrics) in all_results.items():
        epochs = metrics['epochs']
        ax.plot(epochs, metrics['val_mae'], label=config_name, linewidth=2)
    ax.set_title('Validation MAE Progress', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning rate schedules
    ax = axes[1, 0]
    for config_name, (model, metrics) in all_results.items():
        epochs = metrics['epochs']
        ax.plot(epochs, metrics['learning_rates'], label=config_name, linewidth=2)
    ax.set_title('Learning Rate Schedules', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Final performance comparison
    ax = axes[1, 1]
    config_names = list(all_results.keys())
    best_r2_scores = [metrics['best_val_r2'] for _, (model, metrics) in all_results.items()]

    bars = ax.bar(config_names, best_r2_scores, alpha=0.7, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    ax.set_title('Best Validation R² Comparison', fontweight='bold', fontsize=12)
    ax.set_ylabel('R² Score')
    ax.set_xticklabels(config_names, rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, best_r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Training efficiency
    ax = axes[1, 2]
    total_epochs = [metrics['total_epochs'] for _, (model, metrics) in all_results.items()]

    bars = ax.bar(config_names, total_epochs, alpha=0.7, color='orange')
    ax.set_title('Training Efficiency (Total Epochs)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Epochs')
    ax.set_xticklabels(config_names, rotation=45, ha='right')

    for bar, value in zip(bars, total_epochs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Overfitting analysis
    ax = axes[2, 0]
    for config_name, (model, metrics) in all_results.items():
        epochs = metrics['epochs']
        gap = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
        ax.plot(epochs, gap, label=config_name, linewidth=2)
    ax.set_title('Overfitting Analysis (Val - Train Loss)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Gap')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 8: Best MAE comparison
    ax = axes[2, 1]
    best_mae_scores = [metrics['best_val_mae'] for _, (model, metrics) in all_results.items()]

    bars = ax.bar(config_names, best_mae_scores, alpha=0.7, color='lightcoral')
    ax.set_title('Best Validation MAE Comparison', fontweight='bold', fontsize=12)
    ax.set_ylabel('MAE')
    ax.set_xticklabels(config_names, rotation=45, ha='right')

    # Plot 9: Summary statistics
    ax = axes[2, 2]
    ax.axis('off')

    # Find best performing configuration
    best_config = max(all_results.items(), key=lambda x: x[1][1]['best_val_r2'])
    best_name, (best_model, best_metrics) = best_config

    summary_text = f"🏆 TRAINING RESULTS SUMMARY\n\n"
    summary_text += f"Best Configuration: {best_name}\n"
    summary_text += f"Best R² Score: {best_metrics['best_val_r2']:.4f}\n"
    summary_text += f"Best MAE: {best_metrics['best_val_mae']:.4f}\n"
    summary_text += f"Training Epochs: {best_metrics['total_epochs']}\n\n"

    summary_text += f"All Configurations:\n"
    for config_name, (model, metrics) in all_results.items():
        summary_text += f"• {config_name}: R²={metrics['best_val_r2']:.3f}\n"

    summary_text += f"\n🎯 Performance Goals:\n"
    if best_metrics['best_val_r2'] > 0.8:
        summary_text += f"✅ Excellent (R² > 0.8)\n"
    elif best_metrics['best_val_r2'] > 0.7:
        summary_text += f"✅ Good (R² > 0.7)\n"
    elif best_metrics['best_val_r2'] > 0.6:
        summary_text += f"⚠️ Acceptable (R² > 0.6)\n"
    else:
        summary_text += f"❌ Needs Improvement\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('output/training/plots/comprehensive_training_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Comprehensive training comparison saved")
    return best_name, best_metrics

def test_models(all_results, X_test, y_test, scaler_x, scaler_y):
    """Test all trained models and create test visualizations"""
    print("🧪 Testing all trained models...")

    test_results = {}

    for config_name, (model, train_metrics) in all_results.items():
        print(f"  Testing {config_name}...")

        # Make predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            predictions_scaled = model(X_tensor).numpy()

        # Inverse transform predictions to original scale
        predictions = scaler_y.inverse_transform(predictions_scaled)
        actual = scaler_y.inverse_transform(y_test)

        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)

        # Per-target metrics
        target_names = ['Modal', 'Profit', 'Rugi']
        per_target_metrics = {}

        for i, target in enumerate(target_names):
            target_mae = mean_absolute_error(actual[:, i], predictions[:, i])
            target_r2 = r2_score(actual[:, i], predictions[:, i])
            per_target_metrics[target] = {'mae': target_mae, 'r2': target_r2}

        # Business constraints check
        modal_positive = np.all(predictions[:, 0] >= 0)
        rugi_positive = np.all(predictions[:, 2] >= 0)

        test_results[config_name] = {
            'overall_metrics': {
                'mae': mae,
                'r2': r2,
                'mse': mse,
                'rmse': rmse
            },
            'per_target_metrics': per_target_metrics,
            'constraints_passed': {
                'modal_positive': modal_positive,
                'rugi_positive': rugi_positive
            },
            'predictions': predictions,
            'actual': actual
        }

        print(f"    R²: {r2:.4f}, MAE: {mae:.2f}, Constraints: {'✅' if modal_positive and rugi_positive else '❌'}")

    # Create test comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: R² comparison
    ax = axes[0, 0]
    config_names = list(test_results.keys())
    r2_scores = [results['overall_metrics']['r2'] for results in test_results.values()]

    bars = ax.bar(config_names, r2_scores, alpha=0.7, color='lightgreen')
    ax.set_title('Test R² Score Comparison', fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_xticklabels(config_names, rotation=45, ha='right')

    for bar, value in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: MAE comparison
    ax = axes[0, 1]
    mae_scores = [results['overall_metrics']['mae'] for results in test_results.values()]

    bars = ax.bar(config_names, mae_scores, alpha=0.7, color='lightcoral')
    ax.set_title('Test MAE Comparison', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xticklabels(config_names, rotation=45, ha='right')

    # Plot 3: Per-target R² heatmap
    ax = axes[0, 2]
    target_names = ['Modal', 'Profit', 'Rugi']
    heatmap_data = []

    for config_name in config_names:
        row = [test_results[config_name]['per_target_metrics'][target]['r2'] for target in target_names]
        heatmap_data.append(row)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Per-Target R² Scores', fontweight='bold')
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names)

    # Add text annotations
    for i in range(len(config_names)):
        for j in range(len(target_names)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')

    plt.colorbar(im, ax=ax)

    # Plot 4-6: Prediction vs Actual for best model
    best_config = max(test_results.items(), key=lambda x: x[1]['overall_metrics']['r2'])
    best_name, best_results = best_config

    for i, target in enumerate(target_names):
        ax = axes[1, i]

        actual_target = best_results['actual'][:, i]
        pred_target = best_results['predictions'][:, i]

        ax.scatter(actual_target, pred_target, alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(actual_target.min(), pred_target.min())
        max_val = max(actual_target.max(), pred_target.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        r2_target = best_results['per_target_metrics'][target]['r2']
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{target} - Best Model ({best_name})\nR² = {r2_target:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/training/plots/comprehensive_test_results.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Test results visualization saved")
    return test_results, best_name

def main():
    """Main execution function"""
    print("🚀 OPTIMIZED TRAINING DEMONSTRATION")
    print("=" * 80)

    # Load processed data
    X_scaled, y_scaled, scaler_x, scaler_y = load_processed_data()

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"📊 Data splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

    # Define different training configurations
    configs = {
        'conservative': {
            'name': 'conservative',
            'hidden_sizes': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 150,
            'dropout': 0.2,
            'weight_decay': 1e-5,
            'scheduler': 'plateau'
        },
        'balanced': {
            'name': 'balanced',
            'hidden_sizes': [128, 64, 32],
            'learning_rate': 0.003,
            'batch_size': 64,
            'epochs': 200,
            'dropout': 0.3,
            'weight_decay': 1e-4,
            'scheduler': 'plateau'
        },
        'aggressive': {
            'name': 'aggressive',
            'hidden_sizes': [256, 128, 64],
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 250,
            'dropout': 0.4,
            'weight_decay': 1e-3,
            'scheduler': 'cosine'
        },
        'experimental': {
            'name': 'experimental',
            'hidden_sizes': [192, 96, 48, 24],
            'learning_rate': 0.005,
            'batch_size': 96,
            'epochs': 300,
            'dropout': 0.35,
            'weight_decay': 5e-4,
            'scheduler': 'plateau'
        }
    }

    # Train all configurations
    all_results = {}

    for config_name, config in configs.items():
        print(f"\n🎯 Training {config_name} configuration...")
        model, metrics = train_optimized_model(X_train, y_train, X_val, y_val, config)
        all_results[config_name] = (model, metrics)

    # Create comprehensive visualizations
    best_name, best_metrics = create_comprehensive_visualizations(all_results)

    # Test all models
    test_results, best_test_model = test_models(all_results, X_test, y_test, scaler_x, scaler_y)

    # Save comprehensive results
    final_results = {
        'best_training_config': best_name,
        'best_test_model': best_test_model,
        'training_results': {
            name: {
                'best_val_r2': metrics['best_val_r2'],
                'best_val_mae': metrics['best_val_mae'],
                'total_epochs': metrics['total_epochs']
            } for name, (model, metrics) in all_results.items()
        },
        'test_results': {
            name: {
                'r2': results['overall_metrics']['r2'],
                'mae': results['overall_metrics']['mae'],
                'constraints_passed': all(results['constraints_passed'].values())
            } for name, results in test_results.items()
        }
    }

    with open('output/training/comprehensive_optimization_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n🎉 OPTIMIZATION DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print(f"🏆 Best Training Config: {best_name} (R² = {best_metrics['best_val_r2']:.4f})")
    print(f"🥇 Best Test Model: {best_test_model} (R² = {test_results[best_test_model]['overall_metrics']['r2']:.4f})")
    print(f"📁 Results saved in: output/training/")
    print(f"📊 Visualizations: output/training/plots/")
    print(f"🤖 Models: output/training/models/")
    print(f"📋 Logs: output/training/logs/")

    return final_results

if __name__ == "__main__":
    results = main()
    print("\n✅ Demonstration completed successfully!")
