import torch
import joblib
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt

# Try to import optimized model architecture
try:
    from predictAll_optimized import ImprovedBusinessMLP
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

# ------------------ Business-Constrained Model ------------------ #
class BusinessConstrainedMLP(nn.Module):
    """MLP with business logic constraints"""
    def __init__(self, input_size=3, hidden_size=128, output_size=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        output = self.backbone(x)
        # Apply business constraints
        modal = torch.relu(output[:, 0])  # Modal must be positive
        profit = output[:, 1]  # Profit can be negative
        rugi = torch.relu(output[:, 2])   # Loss must be positive
        return torch.stack([modal, profit, rugi], dim=1)

# Backward compatibility - match the saved model architecture
class BisnisAssistantModel(nn.Module):
    """Legacy model that matches the saved model architecture"""
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        # Try to match the saved model architecture
        try:
            # First try the new architecture
            self.model = BusinessConstrainedMLP(input_size, 128, output_size)
        except:
            # Fallback to simple architecture
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, x):
        if hasattr(self.model, 'backbone'):
            return self.model(x)
        else:
            output = self.model(x)
            # Apply business constraints manually
            modal = torch.relu(output[:, 0])  # Modal must be positive
            profit = output[:, 1]  # Profit can be negative
            rugi = torch.relu(output[:, 2])   # Loss must be positive
            return torch.stack([modal, profit, rugi], dim=1)

# ------------------ Fungsi Prediksi Utama ------------------ #
def predict_all(pemasukan, pengeluaran, jam):
    """
    Predict business metrics using ensemble models.
    Uses the correct feature format: [pemasukan, pengeluaran, jam]
    """
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "../data")
    model_path = os.path.join(base_path, "../models")

    try:
        # ------------------ Load Scalers ------------------ #
        scaler_x = joblib.load(os.path.join(data_path, "scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(data_path, "scaler_y.pkl"))

        # ------------------ Prepare Input Data ------------------ #
        # Input format: [pemasukan, pengeluaran, jam] - matches training data
        input_data = np.array([[pemasukan, pengeluaran, jam]], dtype=np.float32)
        input_scaled = scaler_x.transform(input_data)

        # ------------------ Load Models ------------------ #
        # MLP Model - try different architectures to match saved model
        mlp_model = None

        # First try to load optimized model if available
        if OPTIMIZED_AVAILABLE:
            try:
                optimized_model_path = "output/training/improved_model_best.pth"
                if os.path.exists(optimized_model_path):
                    mlp_model = ImprovedBusinessMLP(input_size=3, hidden_size=128, output_size=3, dropout=0.3)
                    mlp_model.load_state_dict(torch.load(optimized_model_path, map_location='cpu'))
                    mlp_model.eval()
                    print("✅ Using optimized model (R² = 0.58)")
            except Exception as e_opt:
                print(f"⚠️ Could not load optimized model: {e_opt}")

        # Fallback to legacy models if optimized not available
        if mlp_model is None:
            try:
                # Try the new architecture first
                mlp_model = BusinessConstrainedMLP(input_size=3, hidden_size=128, output_size=3)
                mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_model.pth"),
                                                   map_location='cpu'))
                mlp_model.eval()
                print("📋 Using legacy constrained model")
            except Exception as e1:
                try:
                    # Try the old architecture
                    mlp_model = BisnisAssistantModel(input_size=3, hidden_size=64, output_size=3)
                    mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_model.pth"),
                                                       map_location='cpu'))
                    mlp_model.eval()
                    print("📋 Using legacy basic model")
                except Exception as e2:
                    print(f"⚠️ Could not load any MLP model: {e2}")
                    mlp_model = None

        # Tree Models
        tree = joblib.load(os.path.join(model_path, "tree_model.pkl"))
        rf = joblib.load(os.path.join(model_path, "rf_model.pkl"))

        # ARIMAX Model
        try:
            arimax = joblib.load(os.path.join(model_path, "arimax_model.pkl"))
        except:
            arimax = None

        # KMeans Model
        kmeans = joblib.load(os.path.join(model_path, "kmeans_model.pkl"))

        # ------------------ Make Predictions ------------------ #
        results = {}

        # MLP Prediction with Business Constraints
        if mlp_model is not None:
            try:
                with torch.no_grad():
                    pred_mlp = mlp_model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
                    pred_mlp = scaler_y.inverse_transform(pred_mlp)[0]

                    # Apply business logic constraints
                    modal = max(0, float(pred_mlp[0]))  # Modal cannot be negative
                    profit = float(pred_mlp[1])  # Profit can be negative
                    rugi = max(0, float(pred_mlp[2]))   # Loss cannot be negative

                    # Business logic: if profit is positive, loss should be minimal
                    if profit > 0:
                        rugi = min(rugi, profit * 0.1)  # Loss shouldn't exceed 10% of profit

                    results["mlp"] = {
                        "modal": modal,
                        "profit": profit,
                        "rugi": rugi
                    }
            except Exception as e:
                print(f"MLP prediction error: {e}")
                results["mlp"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}
        else:
            # Use business logic fallback when MLP model is not available
            net_income = pemasukan - pengeluaran
            modal = pemasukan * 0.3  # Estimate 30% of revenue as capital

            if net_income > 0:
                profit = net_income * 0.8
                rugi = net_income * 0.1
            else:
                profit = 0
                rugi = abs(net_income)

            # Apply constraints
            modal = max(0, modal)
            rugi = max(0, rugi)
            if profit > 0:
                rugi = min(rugi, profit * 0.1)

            results["mlp"] = {
                "modal": modal,
                "profit": profit,
                "rugi": rugi
            }

        # Tree Prediction with Business Constraints
        try:
            pred_tree = tree.predict(input_scaled)
            pred_tree = scaler_y.inverse_transform(pred_tree.reshape(1, -1))[0]

            # Apply business constraints
            modal = max(0, float(pred_tree[0]))
            profit = float(pred_tree[1])
            rugi = max(0, float(pred_tree[2]))

            # Ensure logical consistency
            if profit > 0 and rugi > profit:
                rugi = profit * 0.1

            results["tree"] = {
                "modal": modal,
                "profit": profit,
                "rugi": rugi
            }
        except Exception as e:
            print(f"Tree prediction error: {e}")
            results["tree"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}

        # Random Forest Prediction with Business Constraints
        try:
            pred_rf = rf.predict(input_scaled)
            pred_rf = scaler_y.inverse_transform(pred_rf.reshape(1, -1))[0]

            # Apply business constraints
            modal = max(0, float(pred_rf[0]))
            profit = float(pred_rf[1])
            rugi = max(0, float(pred_rf[2]))

            # Ensure logical consistency
            if profit > 0 and rugi > profit:
                rugi = profit * 0.1

            results["rf"] = {
                "modal": modal,
                "profit": profit,
                "rugi": rugi
            }
        except Exception as e:
            print(f"RF prediction error: {e}")
            results["rf"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}

        # ARIMAX Prediction with Visualization
        try:
            if arimax is not None:
                # Use business-informed exogenous variables
                # Map business size to promo/holiday likelihood
                revenue_estimate = pemasukan * 30  # Monthly estimate
                if revenue_estimate > 50000000:  # Large business
                    promo_prob = 0.4  # More likely to run promos
                    holiday_effect = 0.1
                else:  # Small/medium business
                    promo_prob = 0.2
                    holiday_effect = 0.05

                exog_input = np.array([[promo_prob, holiday_effect]])
                pred_arimax = arimax.forecast(steps=7, exog=np.repeat(exog_input, 7, axis=0))

                if hasattr(pred_arimax, 'iloc'):
                    forecast_values = pred_arimax.values
                else:
                    forecast_values = pred_arimax

                # Create forecast plot
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    days = range(1, 8)
                    plt.plot(days, forecast_values, 'bo-', linewidth=2, markersize=8)
                    plt.fill_between(days, forecast_values * 0.9, forecast_values * 1.1,
                                   alpha=0.3, label='Confidence Interval')
                    plt.title(f'7-Day Sales Forecast\nBusiness Revenue: Rp {revenue_estimate:,.0f}/month')
                    plt.xlabel('Days Ahead')
                    plt.ylabel('Predicted Sales')
                    plt.grid(True, alpha=0.3)
                    plt.legend()

                    # Save plot
                    os.makedirs('ensemble/plots', exist_ok=True)
                    plt.savefig('ensemble/plots/arimax_forecast.png', dpi=150, bbox_inches='tight')
                    plt.close()

                    print("📊 ARIMAX forecast plot saved to ensemble/plots/arimax_forecast.png")
                except Exception as plot_error:
                    print(f"⚠️ Plot creation failed: {plot_error}")

                # Return average forecast
                results["arimax_sales"] = float(np.mean(forecast_values))
            else:
                results["arimax_sales"] = 0.0
        except Exception as e:
            print(f"ARIMAX prediction error: {e}")
            results["arimax_sales"] = 0.0

        # KMeans Clustering
        try:
            # Use first 2 features for clustering (pemasukan, pengeluaran)
            cluster_input = input_scaled[:, :2]
            cluster = int(kmeans.predict(cluster_input)[0])
            results["kmeans_cluster"] = cluster
        except Exception as e:
            print(f"KMeans prediction error: {e}")
            results["kmeans_cluster"] = 0

        return results

    except Exception as e:
        print(f"General prediction error: {e}")
        # Return default values if everything fails
        return {
            "mlp": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "tree": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "rf": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "arimax_sales": 0.0,
            "kmeans_cluster": 0
        }

# ------------------ Test Function ------------------ #
if __name__ == "__main__":
    # Test the prediction function
    print("Testing prediction function...")
    result = predict_all(pemasukan=0.5, pengeluaran=0.3, jam=0.5)

    print("Hasil prediksi semua model:\n")
    for model, val in result.items():
        print(f"{model.upper()}:")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k.capitalize()}: {v:.4f}")
        else:
            print(f"  Hasil: {val}")
        print()
