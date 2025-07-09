#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite
Tests trained models for functional minimum goals and performance validation
"""

import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from optimized_training_system import OptimizedMLP, ResidualMLP

class ModelTester:
    """Comprehensive model testing and validation"""

    def __init__(self, output_dir="output/training"):
        self.output_dir = output_dir
        self.test_results = {}

    def load_model(self, model_path, model_type='standard', config=None):
        """Load trained model"""
        if model_type == 'residual':
            model = ResidualMLP(input_size=3, hidden_size=128, output_size=3)
        else:
            hidden_sizes = config.get('hidden_sizes', [64, 32]) if config else [64, 32]
            model = OptimizedMLP(input_size=3, hidden_sizes=hidden_sizes, output_size=3)

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def test_business_constraints(self, model, X_test, y_test, scaler_x, scaler_y):
        """Test business logic constraints"""
        print("🔍 Testing business constraints...")

        # Scale test data
        X_test_scaled = scaler_x.transform(X_test)

        # Make predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            predictions_scaled = model(X_tensor).numpy()

        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(predictions_scaled)

        # Test constraints
        modal_positive = np.all(predictions[:, 0] >= 0)
        rugi_positive = np.all(predictions[:, 2] >= 0)

        # Business logic tests
        reasonable_modal = np.all((predictions[:, 0] >= 1000) & (predictions[:, 0] <= 1e9))
        reasonable_profit = np.all(predictions[:, 1] >= -1e8)
        reasonable_rugi = np.all(predictions[:, 2] <= 1e8)

        constraints_passed = {
            'modal_positive': modal_positive,
            'rugi_positive': rugi_positive,
            'reasonable_modal': reasonable_modal,
            'reasonable_profit': reasonable_profit,
            'reasonable_rugi': reasonable_rugi
        }

        print(f"  ✅ Modal positive: {modal_positive}")
        print(f"  ✅ Rugi positive: {rugi_positive}")
        print(f"  ✅ Reasonable modal range: {reasonable_modal}")
        print(f"  ✅ Reasonable profit range: {reasonable_profit}")
        print(f"  ✅ Reasonable rugi range: {reasonable_rugi}")

        return constraints_passed, predictions

    def test_model_performance(self, model, X_test, y_test, scaler_x, scaler_y, model_name):
        """Test model performance metrics"""
        print(f"📊 Testing performance for {model_name}...")

        # Get predictions
        constraints_passed, predictions = self.test_business_constraints(
            model, X_test, y_test, scaler_x, scaler_y
        )

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)

        # Per-target metrics
        target_names = ['modal', 'profit', 'rugi']
        per_target_metrics = {}

        for i, target in enumerate(target_names):
            target_mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            target_r2 = r2_score(y_test[:, i], predictions[:, i])
            per_target_metrics[target] = {
                'mae': target_mae,
                'r2': target_r2
            }

        # Functional minimum goals check
        functional_goals = {
            'overall_r2_above_0.7': r2 > 0.7,
            'overall_mae_below_threshold': mae < np.mean(np.abs(y_test)) * 0.3,
            'modal_r2_above_0.6': per_target_metrics['modal']['r2'] > 0.6,
            'profit_r2_above_0.5': per_target_metrics['profit']['r2'] > 0.5,
            'rugi_r2_above_0.5': per_target_metrics['rugi']['r2'] > 0.5,
            'all_constraints_passed': all(constraints_passed.values())
        }

        results = {
            'model_name': model_name,
            'overall_metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            },
            'per_target_metrics': per_target_metrics,
            'constraints_passed': constraints_passed,
            'functional_goals': functional_goals,
            'predictions': predictions.tolist(),
            'actual': y_test.tolist()
        }

        # Print results
        print(f"  📈 Overall R²: {r2:.4f}")
        print(f"  📉 Overall MAE: {mae:.2f}")
        print(f"  🎯 Functional goals passed: {sum(functional_goals.values())}/{len(functional_goals)}")

        return results

    def create_test_visualizations(self, results, model_name):
        """Create comprehensive test visualizations"""
        print(f"📊 Creating test visualizations for {model_name}...")

        predictions = np.array(results['predictions'])
        actual = np.array(results['actual'])
        target_names = ['Modal', 'Profit', 'Rugi']

        # Create test results dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1-3: Prediction vs Actual for each target
        for i, target in enumerate(target_names):
            ax = axes[0, i]

            # Scatter plot
            ax.scatter(actual[:, i], predictions[:, i], alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(actual[:, i].min(), predictions[:, i].min())
            max_val = max(actual[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            # Metrics
            r2 = results['per_target_metrics'][target.lower()]['r2']
            mae = results['per_target_metrics'][target.lower()]['mae']

            ax.set_xlabel(f'Actual {target}')
            ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'{target} Predictions\nR² = {r2:.3f}, MAE = {mae:.2f}')
            ax.grid(True, alpha=0.3)

        # Plot 4: Residuals analysis
        ax = axes[1, 0]
        residuals = predictions - actual
        overall_residuals = np.mean(np.abs(residuals), axis=1)

        ax.hist(overall_residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Mean Absolute Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residuals Distribution')
        ax.grid(True, alpha=0.3)

        # Plot 5: Performance metrics bar chart
        ax = axes[1, 1]
        metrics_names = ['Modal R²', 'Profit R²', 'Rugi R²']
        metrics_values = [results['per_target_metrics'][target.lower()]['r2']
                         for target in ['modal', 'profit', 'rugi']]

        bars = ax.bar(metrics_names, metrics_values, alpha=0.7)
        ax.set_ylabel('R² Score')
        ax.set_title('Per-Target Performance')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 6: Test summary
        ax = axes[1, 2]
        ax.axis('off')

        # Create summary text
        summary_text = f"🧪 TEST RESULTS SUMMARY\n\n"
        summary_text += f"Model: {model_name}\n\n"

        summary_text += "📊 Overall Metrics:\n"
        summary_text += f"  • R² Score: {results['overall_metrics']['r2']:.4f}\n"
        summary_text += f"  • MAE: {results['overall_metrics']['mae']:.2f}\n"
        summary_text += f"  • RMSE: {results['overall_metrics']['rmse']:.2f}\n\n"

        summary_text += "🎯 Functional Goals:\n"
        goals_passed = sum(results['functional_goals'].values())
        total_goals = len(results['functional_goals'])
        summary_text += f"  • Passed: {goals_passed}/{total_goals}\n"

        for goal, passed in results['functional_goals'].items():
            status = "✅" if passed else "❌"
            summary_text += f"  {status} {goal.replace('_', ' ').title()}\n"

        summary_text += "\n🔒 Business Constraints:\n"
        constraints_passed = sum(results['constraints_passed'].values())
        total_constraints = len(results['constraints_passed'])
        summary_text += f"  • Passed: {constraints_passed}/{total_constraints}\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'output/training/plots/{model_name}_test_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Test visualization saved: {model_name}_test_results.png")

    def run_comprehensive_tests(self):
        """Run comprehensive tests on all trained models"""
        print("🧪 Starting Comprehensive Model Testing")
        print("=" * 80)

        # Load test data
        from optimized_training_system import load_and_preprocess_data
        X, y = load_and_preprocess_data()

        # Use the same split as training
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"📊 Test data: {X_test.shape[0]} samples")

        # Find all trained models
        models_dir = f"{self.output_dir}/models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

        all_test_results = {}

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace('.pth', '')

            print(f"\n🔍 Testing model: {model_name}")

            try:
                # Determine model type and scaling
                if 'residual' in model_name:
                    model_type = 'residual'
                else:
                    model_type = 'standard'

                # Find corresponding scaler
                scaling_type = 'standard'  # Default
                for scaling in ['standard', 'minmax', 'robust']:
                    if scaling in model_name:
                        scaling_type = scaling
                        break

                scaler_x_path = f"{models_dir}/scaler_x_{scaling_type}.pkl"
                scaler_y_path = f"{models_dir}/scaler_y_{scaling_type}.pkl"

                if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
                    scaler_x = joblib.load(scaler_x_path)
                    scaler_y = joblib.load(scaler_y_path)

                    # Load and test model
                    model = self.load_model(model_path, model_type)
                    results = self.test_model_performance(
                        model, X_test, y_test, scaler_x, scaler_y, model_name
                    )

                    # Create visualizations
                    self.create_test_visualizations(results, model_name)

                    all_test_results[model_name] = results

                else:
                    print(f"  ❌ Scalers not found for {model_name}")

            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {str(e)}")

        # Save comprehensive test results
        with open(f'{self.output_dir}/logs/comprehensive_test_results.json', 'w') as f:
            json.dump(all_test_results, f, indent=2)

        # Find best performing model
        best_model = None
        best_score = -1

        for model_name, results in all_test_results.items():
            score = results['overall_metrics']['r2']
            if score > best_score:
                best_score = score
                best_model = model_name

        print("\n🏆 TESTING COMPLETED!")
        print("=" * 80)
        print(f"🥇 Best Model: {best_model}")
        print(f"🎯 Best R² Score: {best_score:.4f}")
        print(f"📁 Test results saved in: {self.output_dir}/logs/")
        print(f"📊 Test visualizations saved in: {self.output_dir}/plots/")

        return all_test_results, best_model, best_score

if __name__ == "__main__":
    tester = ModelTester()
    test_results, best_model, best_score = tester.run_comprehensive_tests()
    print(f"\n✅ Testing completed successfully!")
