#!/usr/bin/env python3
"""
Optimized Training System with Advanced Strategies
Multiple optimization approaches for achieving functional minimum goals
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs("output/training", exist_ok=True)
os.makedirs("output/training/models", exist_ok=True)
os.makedirs("output/training/plots", exist_ok=True)
os.makedirs("output/training/logs", exist_ok=True)

# ================= TRAINING CONFIGURATIONS ================= #

class TrainingConfig:
    """Centralized training configuration with multiple strategies"""

    STRATEGIES = {
        'conservative': {
            'mlp': {
                'hidden_sizes': [64, 32],
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 200,
                'dropout': 0.2,
                'weight_decay': 1e-5,
                'scheduler': 'plateau'
            },
            'tree': {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'n_estimators': 100
            }
        },
        'aggressive': {
            'mlp': {
                'hidden_sizes': [128, 64, 32],
                'learning_rate': 0.01,
                'batch_size': 64,
                'epochs': 500,
                'dropout': 0.3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine'
            },
            'tree': {
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'n_estimators': 200
            }
        },
        'balanced': {
            'mlp': {
                'hidden_sizes': [96, 48, 24],
                'learning_rate': 0.005,
                'batch_size': 48,
                'epochs': 300,
                'dropout': 0.25,
                'weight_decay': 5e-5,
                'scheduler': 'step'
            },
            'tree': {
                'max_depth': 15,
                'min_samples_split': 8,
                'min_samples_leaf': 3,
                'n_estimators': 150
            }
        },
        'experimental': {
            'mlp': {
                'hidden_sizes': [256, 128, 64, 32],
                'learning_rate': 0.003,
                'batch_size': 128,
                'epochs': 400,
                'dropout': 0.4,
                'weight_decay': 1e-3,
                'scheduler': 'plateau'
            },
            'tree': {
                'max_depth': 25,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'n_estimators': 300
            }
        }
    }

# ================= ADVANCED MODEL ARCHITECTURES ================= #

class OptimizedMLP(nn.Module):
    """Advanced MLP with configurable architecture and business constraints"""

    def __init__(self, input_size=3, hidden_sizes=[64, 32], output_size=3, dropout=0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
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

class ResidualMLP(nn.Module):
    """MLP with residual connections for better gradient flow"""

    def __init__(self, input_size=3, hidden_size=128, output_size=3, dropout=0.2):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_size, dropout) for _ in range(3)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def _make_residual_block(self, hidden_size, dropout):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

    def forward(self, x):
        x = self.input_projection(x)

        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)  # Residual connection

        x = self.dropout(x)
        output = self.output_layer(x)

        # Apply business constraints
        modal = torch.relu(output[:, 0])
        profit = output[:, 1]
        rugi = torch.relu(output[:, 2])
        return torch.stack([modal, profit, rugi], dim=1)

# ================= DATA PREPROCESSING ================= #

def load_and_preprocess_data():
    """Load and preprocess business data with multiple scaling strategies"""
    print("📊 Loading and preprocessing data...")

    # Load data
    df = pd.read_csv("dataset/csv/business_owner_dataset_extended.csv")

    # Feature engineering
    features = ['pemasukan', 'pengeluaran', 'jam']
    targets = ['modal', 'profit', 'rugi']

    X = df[features].values
    y = df[targets].values

    # Remove outliers using IQR method
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    X = X[mask]
    y = y[mask]

    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Outliers removed: {np.sum(~mask)} samples")

    return X, y

def create_scalers(X, y, scaler_type='standard'):
    """Create and fit scalers with different strategies"""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    scaler_x = scalers[scaler_type]
    scaler_y = scalers[scaler_type]

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_x, scaler_y

# ================= ADVANCED TRAINING FUNCTIONS ================= #

class TrainingLogger:
    """Advanced training logger with detailed metrics tracking"""

    def __init__(self, strategy_name, model_type):
        self.strategy_name = strategy_name
        self.model_type = model_type
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rates': [],
            'epochs': []
        }
        self.best_metrics = {}

    def log_epoch(self, epoch, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2, lr):
        """Log metrics for an epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['val_mae'].append(val_mae)
        self.metrics['train_r2'].append(train_r2)
        self.metrics['val_r2'].append(val_r2)
        self.metrics['learning_rates'].append(lr)

    def save_metrics(self):
        """Save metrics to JSON file"""
        filename = f"output/training/logs/{self.strategy_name}_{self.model_type}_metrics.json"
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save best metrics
        self.best_metrics = {
            'best_val_loss': min(self.metrics['val_loss']),
            'best_val_mae': min(self.metrics['val_mae']),
            'best_val_r2': max(self.metrics['val_r2']),
            'final_train_loss': self.metrics['train_loss'][-1],
            'total_epochs': len(self.metrics['epochs'])
        }

        best_filename = f"output/training/logs/{self.strategy_name}_{self.model_type}_best.json"
        with open(best_filename, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)

def train_optimized_mlp(X_train, y_train, X_val, y_val, strategy='balanced', model_type='standard'):
    """Train MLP with advanced optimization strategies"""
    print(f"🧠 Training {model_type} MLP with {strategy} strategy...")

    config = TrainingConfig.STRATEGIES[strategy]['mlp']
    logger = TrainingLogger(strategy, f"mlp_{model_type}")

    # Create model
    if model_type == 'residual':
        model = ResidualMLP(
            input_size=3,
            hidden_size=config['hidden_sizes'][0],
            output_size=3,
            dropout=config['dropout']
        )
    else:
        model = OptimizedMLP(
            input_size=3,
            hidden_sizes=config['hidden_sizes'],
            output_size=3,
            dropout=config['dropout']
        )

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Setup scheduler
    if config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)
    elif config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:  # step
        scheduler = StepLR(optimizer, step_size=50, gamma=0.7)

    # Create data loaders
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

    # Training loop with advanced monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

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

        train_mae = mean_absolute_error(train_true, train_pred)
        val_mae = mean_absolute_error(val_true, val_pred)
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)

        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logger.log_epoch(epoch, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2, current_lr)

        # Update scheduler
        if config['scheduler'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"output/training/models/{strategy}_{model_type}_mlp_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Progress reporting
        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, "
                  f"Val MAE {val_mae:.4f}, Val R² {val_r2:.4f}, LR {current_lr:.6f}")

    # Save final metrics and model
    logger.save_metrics()
    torch.save(model.state_dict(), f"output/training/models/{strategy}_{model_type}_mlp_final.pth")

    print(f"✅ {model_type.title()} MLP training completed. Best validation loss: {best_val_loss:.6f}")
    return model, logger

def train_optimized_tree_models(X_train, y_train, X_val, y_val, strategy='balanced'):
    """Train tree-based models with advanced configurations"""
    print(f"🌳 Training tree models with {strategy} strategy...")

    config = TrainingConfig.STRATEGIES[strategy]['tree']
    results = {}

    models = {
        'decision_tree': DecisionTreeRegressor(
            random_state=42,
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features='sqrt'
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=config['n_estimators'],
            random_state=42,
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features='sqrt',
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=config['n_estimators'],
            random_state=42,
            max_depth=config['max_depth'],
            learning_rate=0.1,
            subsample=0.8
        )
    }

    for name, model in models.items():
        print(f"  Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)

        # Store results
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse
        }

        # Save model
        joblib.dump(model, f"output/training/models/{strategy}_{name}.pkl")

        print(f"    ✅ {name}: Val MAE {val_mae:.4f}, Val R² {val_r2:.4f}")

    # Save results
    metrics_summary = {name: {k: v for k, v in result.items() if k != 'model'}
                      for name, result in results.items()}

    with open(f"output/training/logs/{strategy}_tree_models_results.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"✅ Tree models training completed for {strategy} strategy")
    return results

# ================= VISUALIZATION FUNCTIONS ================= #

def create_training_visualizations(loggers, strategy_name):
    """Create comprehensive training visualizations"""
    print(f"📊 Creating training visualizations for {strategy_name}...")

    # Set style
    plt.style.use('default')
    # sns.set_palette("husl")  # Optional seaborn styling

    # Create comprehensive training dashboard
    fig = plt.figure(figsize=(20, 15))

    # Plot 1: Loss curves comparison
    ax1 = plt.subplot(3, 3, 1)
    for logger in loggers:
        plt.plot(logger.metrics['epochs'], logger.metrics['train_loss'],
                label=f'{logger.model_type} Train', alpha=0.7)
        plt.plot(logger.metrics['epochs'], logger.metrics['val_loss'],
                label=f'{logger.model_type} Val', linestyle='--', alpha=0.7)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: MAE comparison
    ax2 = plt.subplot(3, 3, 2)
    for logger in loggers:
        plt.plot(logger.metrics['epochs'], logger.metrics['val_mae'],
                label=f'{logger.model_type}', linewidth=2)
    plt.title('Validation MAE Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: R² Score comparison
    ax3 = plt.subplot(3, 3, 3)
    for logger in loggers:
        plt.plot(logger.metrics['epochs'], logger.metrics['val_r2'],
                label=f'{logger.model_type}', linewidth=2)
    plt.title('Validation R² Score Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Learning rate schedules
    ax4 = plt.subplot(3, 3, 4)
    for logger in loggers:
        plt.plot(logger.metrics['epochs'], logger.metrics['learning_rates'],
                label=f'{logger.model_type}', linewidth=2)
    plt.title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Final metrics comparison (bar chart)
    ax5 = plt.subplot(3, 3, 5)
    model_names = [logger.model_type for logger in loggers]
    final_val_losses = [logger.metrics['val_loss'][-1] for logger in loggers]
    bars = plt.bar(model_names, final_val_losses, alpha=0.7)
    plt.title('Final Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, final_val_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_val_losses)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Training efficiency (epochs to convergence)
    ax6 = plt.subplot(3, 3, 6)
    epochs_to_best = []
    for logger in loggers:
        best_val_idx = np.argmin(logger.metrics['val_loss'])
        epochs_to_best.append(logger.metrics['epochs'][best_val_idx])

    bars = plt.bar(model_names, epochs_to_best, alpha=0.7, color='orange')
    plt.title('Training Efficiency (Epochs to Best)', fontsize=14, fontweight='bold')
    plt.ylabel('Epochs')
    plt.xticks(rotation=45)

    for bar, value in zip(bars, epochs_to_best):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(epochs_to_best)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Overfitting analysis
    ax7 = plt.subplot(3, 3, 7)
    for logger in loggers:
        train_val_gap = np.array(logger.metrics['val_loss']) - np.array(logger.metrics['train_loss'])
        plt.plot(logger.metrics['epochs'], train_val_gap,
                label=f'{logger.model_type}', linewidth=2)
    plt.title('Overfitting Analysis (Val - Train Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot 8: Best metrics summary
    ax8 = plt.subplot(3, 3, 8)
    best_r2_scores = [max(logger.metrics['val_r2']) for logger in loggers]
    bars = plt.bar(model_names, best_r2_scores, alpha=0.7, color='green')
    plt.title('Best Validation R² Scores', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    for bar, value in zip(bars, best_r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 9: Training summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"Training Summary - {strategy_name.title()} Strategy\n\n"
    for logger in loggers:
        best_val_loss = min(logger.metrics['val_loss'])
        best_val_r2 = max(logger.metrics['val_r2'])
        total_epochs = len(logger.metrics['epochs'])

        summary_text += f"{logger.model_type.title()}:\n"
        summary_text += f"  • Best Val Loss: {best_val_loss:.6f}\n"
        summary_text += f"  • Best Val R²: {best_val_r2:.4f}\n"
        summary_text += f"  • Total Epochs: {total_epochs}\n\n"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'output/training/plots/{strategy_name}_training_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training dashboard saved: {strategy_name}_training_dashboard.png")

def create_model_comparison_plots(all_results):
    """Create comprehensive model comparison across all strategies"""
    print("📊 Creating model comparison plots...")

    # Prepare data for comparison
    strategies = list(all_results.keys())
    model_types = ['mlp_standard', 'mlp_residual']

    # Create comparison dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Best validation loss comparison
    ax = axes[0, 0]
    strategy_names = []
    best_losses = []

    for strategy in strategies:
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                strategy_names.append(f"{strategy}_{model_type}")
                best_losses.append(min(logger.metrics['val_loss']))

    bars = ax.bar(range(len(strategy_names)), best_losses, alpha=0.7)
    ax.set_title('Best Validation Loss Across Strategies', fontweight='bold')
    ax.set_ylabel('Validation Loss')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, best_losses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(best_losses)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Training efficiency comparison
    ax = axes[0, 1]
    epochs_to_best = []

    for strategy in strategies:
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                best_idx = np.argmin(logger.metrics['val_loss'])
                epochs_to_best.append(logger.metrics['epochs'][best_idx])

    bars = ax.bar(range(len(strategy_names)), epochs_to_best, alpha=0.7, color='orange')
    ax.set_title('Training Efficiency (Epochs to Best)', fontweight='bold')
    ax.set_ylabel('Epochs')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')

    # Plot 3: Best R² scores
    ax = axes[0, 2]
    best_r2_scores = []

    for strategy in strategies:
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                best_r2_scores.append(max(logger.metrics['val_r2']))

    bars = ax.bar(range(len(strategy_names)), best_r2_scores, alpha=0.7, color='green')
    ax.set_title('Best R² Scores', fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')

    # Plot 4: Tree models comparison
    ax = axes[1, 0]
    tree_results = []
    tree_names = []

    for strategy in strategies:
        if 'tree_results' in all_results[strategy]:
            for model_name, metrics in all_results[strategy]['tree_results'].items():
                tree_names.append(f"{strategy}_{model_name}")
                tree_results.append(metrics['val_r2'])

    if tree_results:
        bars = ax.bar(range(len(tree_names)), tree_results, alpha=0.7, color='brown')
        ax.set_title('Tree Models R² Comparison', fontweight='bold')
        ax.set_ylabel('R² Score')
        ax.set_xticks(range(len(tree_names)))
        ax.set_xticklabels(tree_names, rotation=45, ha='right')

    # Plot 5: Overall performance heatmap
    ax = axes[1, 1]

    # Create performance matrix
    performance_matrix = []
    strategy_labels = []
    model_labels = []

    for strategy in strategies:
        strategy_row = []
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                # Normalize score (higher is better)
                r2_score = max(logger.metrics['val_r2'])
                strategy_row.append(r2_score)
            else:
                strategy_row.append(0)

        if strategy_row:
            performance_matrix.append(strategy_row)
            strategy_labels.append(strategy)

    if performance_matrix:
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_title('Performance Heatmap (R² Scores)', fontweight='bold')
        ax.set_xticks(range(len(model_types)))
        ax.set_xticklabels(model_types)
        ax.set_yticks(range(len(strategy_labels)))
        ax.set_yticklabels(strategy_labels)

        # Add text annotations
        for i in range(len(strategy_labels)):
            for j in range(len(model_types)):
                if j < len(performance_matrix[i]):
                    text = ax.text(j, i, f'{performance_matrix[i][j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')

        plt.colorbar(im, ax=ax)

    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Find best performing model
    best_strategy = None
    best_model = None
    best_score = -1

    for strategy in strategies:
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                score = max(logger.metrics['val_r2'])
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    best_model = model_type

    summary_text = "🏆 TRAINING RESULTS SUMMARY\n\n"
    summary_text += f"Best Performing Model:\n"
    summary_text += f"  Strategy: {best_strategy}\n"
    summary_text += f"  Model: {best_model}\n"
    summary_text += f"  R² Score: {best_score:.4f}\n\n"

    summary_text += f"Total Strategies Tested: {len(strategies)}\n"
    summary_text += f"Total Models Trained: {len(strategy_names)}\n\n"

    summary_text += "Strategy Performance Ranking:\n"
    strategy_scores = {}
    for strategy in strategies:
        scores = []
        for model_type in model_types:
            if f'{model_type}_logger' in all_results[strategy]:
                logger = all_results[strategy][f'{model_type}_logger']
                scores.append(max(logger.metrics['val_r2']))
        if scores:
            strategy_scores[strategy] = np.mean(scores)

    ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (strategy, score) in enumerate(ranked_strategies, 1):
        summary_text += f"  {i}. {strategy}: {score:.4f}\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.savefig('output/training/plots/comprehensive_model_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Comprehensive model comparison saved")
    return best_strategy, best_model, best_score

# ================= MAIN EXECUTION FUNCTIONS ================= #

def run_comprehensive_training():
    """Run comprehensive training with all strategies"""
    print("🚀 Starting Comprehensive Model Training System")
    print("=" * 80)

    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"📊 Data splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

    # Test different scaling strategies
    scaling_strategies = ['standard', 'minmax', 'robust']
    training_strategies = ['conservative', 'balanced', 'aggressive', 'experimental']

    all_results = {}

    for scaling in scaling_strategies:
        print(f"\n🔄 Testing {scaling} scaling...")

        # Scale data
        X_train_scaled, y_train_scaled, scaler_x, scaler_y = create_scalers(X_train, y_train, scaling)
        X_val_scaled = scaler_x.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)
        X_test_scaled = scaler_x.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        # Save scalers
        joblib.dump(scaler_x, f"output/training/models/scaler_x_{scaling}.pkl")
        joblib.dump(scaler_y, f"output/training/models/scaler_y_{scaling}.pkl")

        for strategy in training_strategies:
            print(f"\n🎯 Training with {strategy} strategy using {scaling} scaling...")
            strategy_key = f"{scaling}_{strategy}"
            all_results[strategy_key] = {}

            # Train MLP models
            loggers = []

            # Standard MLP
            model_std, logger_std = train_optimized_mlp(
                X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                strategy=strategy, model_type='standard'
            )
            loggers.append(logger_std)
            all_results[strategy_key]['mlp_standard_logger'] = logger_std

            # Residual MLP
            model_res, logger_res = train_optimized_mlp(
                X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                strategy=strategy, model_type='residual'
            )
            loggers.append(logger_res)
            all_results[strategy_key]['mlp_residual_logger'] = logger_res

            # Train tree models
            tree_results = train_optimized_tree_models(
                X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                strategy=strategy
            )
            all_results[strategy_key]['tree_results'] = tree_results

            # Create visualizations for this strategy
            create_training_visualizations(loggers, strategy_key)

    # Create comprehensive comparison
    best_strategy, best_model, best_score = create_model_comparison_plots(all_results)

    # Save comprehensive results
    results_summary = {
        'best_configuration': {
            'strategy': best_strategy,
            'model': best_model,
            'score': best_score
        },
        'all_results': {
            strategy: {
                model_key: {
                    'best_val_loss': min(logger.metrics['val_loss']) if hasattr(logger, 'metrics') else None,
                    'best_val_r2': max(logger.metrics['val_r2']) if hasattr(logger, 'metrics') else None,
                    'total_epochs': len(logger.metrics['epochs']) if hasattr(logger, 'metrics') else None
                } for model_key, logger in strategy_results.items()
                if hasattr(logger, 'metrics')
            } for strategy, strategy_results in all_results.items()
        }
    }

    with open('output/training/logs/comprehensive_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n🎉 COMPREHENSIVE TRAINING COMPLETED!")
    print("=" * 80)
    print(f"🏆 Best Configuration: {best_strategy} - {best_model}")
    print(f"🎯 Best R² Score: {best_score:.4f}")
    print(f"📁 All results saved in: output/training/")
    print(f"📊 Visualizations saved in: output/training/plots/")
    print(f"📋 Logs saved in: output/training/logs/")

    return all_results, best_strategy, best_model, best_score

if __name__ == "__main__":
    # Run comprehensive training
    results, best_strategy, best_model, best_score = run_comprehensive_training()

    print(f"\n✅ Training completed successfully!")
    print(f"🔍 Check output/training/ for detailed results and visualizations")
