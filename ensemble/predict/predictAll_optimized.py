import torch
import joblib
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------ Optimized Model Architecture ------------------ #
class ImprovedBusinessMLP(nn.Module):
    """Improved MLP with business constraints - matches our optimized model"""
    
    def __init__(self, input_size=3, hidden_size=128, output_size=3, dropout=0.3):
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

# ------------------ Legacy Model for Backward Compatibility ------------------ #
class BusinessConstrainedMLP(nn.Module):
    """Legacy MLP with business logic constraints"""
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

def predict_all_optimized(pemasukan, pengeluaran, jam):
    """
    Enhanced prediction function using optimized models
    
    Args:
        pemasukan (float): Income/revenue
        pengeluaran (float): Expenses
        jam (float): Operating hours (0-1 normalized)
    
    Returns:
        dict: Predictions from all models including optimized MLP
    """
    
    try:
        print("🚀 Using Optimized Ensemble Prediction System")
        print("-" * 50)
        
        # ------------------ Load Models and Scalers ------------------ #
        
        # Try to load optimized model first
        optimized_model = None
        optimized_scaler_x = None
        optimized_scaler_y = None
        
        try:
            # Load our new optimized model
            optimized_model = ImprovedBusinessMLP(input_size=3, hidden_size=128, output_size=3, dropout=0.3)
            optimized_model.load_state_dict(torch.load("output/training/improved_model_best.pth", map_location='cpu'))
            optimized_model.eval()
            
            # Load optimized scalers
            optimized_scaler_x = joblib.load("ensemble/data/scaler_x.pkl")
            optimized_scaler_y = joblib.load("ensemble/data/scaler_y.pkl")
            
            print("✅ Optimized model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load optimized model: {e}")
            print("📋 Falling back to legacy models...")
        
        # Load legacy models and scalers
        model_path = "ensemble/models"
        
        # Legacy scalers
        try:
            scaler_x = joblib.load(os.path.join("ensemble/data", "scaler_x.pkl"))
            scaler_y = joblib.load(os.path.join("ensemble/data", "scaler_y.pkl"))
        except:
            print("⚠️ Could not load legacy scalers")
            scaler_x = optimized_scaler_x
            scaler_y = optimized_scaler_y
        
        # Legacy MLP Model
        legacy_mlp_model = None
        try:
            legacy_mlp_model = BusinessConstrainedMLP(input_size=3, hidden_size=128, output_size=3)
            legacy_mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_model.pth"), map_location='cpu'))
            legacy_mlp_model.eval()
            print("✅ Legacy MLP model loaded")
        except Exception as e:
            print(f"⚠️ Could not load legacy MLP model: {e}")
        
        # Tree Models
        try:
            tree = joblib.load(os.path.join(model_path, "tree_model.pkl"))
            rf = joblib.load(os.path.join(model_path, "rf_model.pkl"))
            print("✅ Tree models loaded")
        except Exception as e:
            print(f"⚠️ Could not load tree models: {e}")
            tree = None
            rf = None
        
        # ARIMAX Model
        try:
            arimax = joblib.load(os.path.join(model_path, "arimax_model.pkl"))
            print("✅ ARIMAX model loaded")
        except:
            arimax = None
            print("⚠️ ARIMAX model not available")
        
        # KMeans Model
        try:
            kmeans = joblib.load(os.path.join(model_path, "kmeans_model.pkl"))
            print("✅ KMeans model loaded")
        except:
            kmeans = None
            print("⚠️ KMeans model not available")
        
        # ------------------ Prepare Input Data ------------------ #
        input_data = np.array([[pemasukan, pengeluaran, jam]])
        
        # Use optimized scaler if available, otherwise legacy
        active_scaler_x = optimized_scaler_x if optimized_scaler_x is not None else scaler_x
        active_scaler_y = optimized_scaler_y if optimized_scaler_y is not None else scaler_y
        
        input_scaled = active_scaler_x.transform(input_data)
        
        print(f"\n📊 Input Data:")
        print(f"  • Pemasukan: Rp {pemasukan:,.0f}")
        print(f"  • Pengeluaran: Rp {pengeluaran:,.0f}")
        print(f"  • Jam Operasi: {jam:.2f}")
        
        # ------------------ Make Predictions ------------------ #
        results = {}
        
        # 1. OPTIMIZED MLP PREDICTION (Priority)
        if optimized_model is not None:
            try:
                with torch.no_grad():
                    pred_optimized = optimized_model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
                    pred_optimized = active_scaler_y.inverse_transform(pred_optimized)[0]
                    
                    # Business constraints are already applied in the model
                    modal = float(pred_optimized[0])
                    profit = float(pred_optimized[1])
                    rugi = float(pred_optimized[2])
                    
                    results["optimized_mlp"] = {
                        "modal": modal,
                        "profit": profit,
                        "rugi": rugi
                    }
                    
                    print(f"\n🧠 Optimized MLP Prediction:")
                    print(f"  • Modal: Rp {modal:,.0f}")
                    print(f"  • Profit: Rp {profit:,.0f}")
                    print(f"  • Rugi: Rp {rugi:,.0f}")
                    
            except Exception as e:
                print(f"❌ Optimized MLP prediction error: {e}")
                results["optimized_mlp"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}
        
        # 2. LEGACY MLP PREDICTION
        if legacy_mlp_model is not None:
            try:
                with torch.no_grad():
                    pred_legacy = legacy_mlp_model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
                    pred_legacy = active_scaler_y.inverse_transform(pred_legacy)[0]
                    
                    modal = max(0, float(pred_legacy[0]))
                    profit = float(pred_legacy[1])
                    rugi = max(0, float(pred_legacy[2]))
                    
                    results["legacy_mlp"] = {
                        "modal": modal,
                        "profit": profit,
                        "rugi": rugi
                    }
                    
                    print(f"\n🔄 Legacy MLP Prediction:")
                    print(f"  • Modal: Rp {modal:,.0f}")
                    print(f"  • Profit: Rp {profit:,.0f}")
                    print(f"  • Rugi: Rp {rugi:,.0f}")
                    
            except Exception as e:
                print(f"❌ Legacy MLP prediction error: {e}")
                results["legacy_mlp"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}
        
        # 3. TREE MODEL PREDICTIONS
        if tree is not None:
            try:
                pred_tree = tree.predict(input_scaled)
                pred_tree = active_scaler_y.inverse_transform(pred_tree.reshape(1, -1))[0]
                
                modal = max(0, float(pred_tree[0]))
                profit = float(pred_tree[1])
                rugi = max(0, float(pred_tree[2]))
                
                results["decision_tree"] = {
                    "modal": modal,
                    "profit": profit,
                    "rugi": rugi
                }
                
                print(f"\n🌳 Decision Tree Prediction:")
                print(f"  • Modal: Rp {modal:,.0f}")
                print(f"  • Profit: Rp {profit:,.0f}")
                print(f"  • Rugi: Rp {rugi:,.0f}")
                
            except Exception as e:
                print(f"❌ Decision Tree prediction error: {e}")
                results["decision_tree"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}
        
        # 4. RANDOM FOREST PREDICTIONS
        if rf is not None:
            try:
                pred_rf = rf.predict(input_scaled)
                pred_rf = active_scaler_y.inverse_transform(pred_rf.reshape(1, -1))[0]
                
                modal = max(0, float(pred_rf[0]))
                profit = float(pred_rf[1])
                rugi = max(0, float(pred_rf[2]))
                
                results["random_forest"] = {
                    "modal": modal,
                    "profit": profit,
                    "rugi": rugi
                }
                
                print(f"\n🌲 Random Forest Prediction:")
                print(f"  • Modal: Rp {modal:,.0f}")
                print(f"  • Profit: Rp {profit:,.0f}")
                print(f"  • Rugi: Rp {rugi:,.0f}")
                
            except Exception as e:
                print(f"❌ Random Forest prediction error: {e}")
                results["random_forest"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}
        
        # 5. ARIMAX SALES FORECAST
        if arimax is not None:
            try:
                # Create forecast for next 7 days
                forecast_steps = 7
                exog_forecast = np.array([[1, 0]] * forecast_steps)  # Assume promo=1, holiday=0
                
                forecast = arimax.forecast(steps=forecast_steps, exog=exog_forecast)
                forecast_values = forecast if hasattr(forecast, '__iter__') else [forecast]
                
                avg_forecast = float(np.mean(forecast_values))
                results["arimax_sales_forecast"] = avg_forecast
                
                print(f"\n📈 ARIMAX Sales Forecast (7-day avg): Rp {avg_forecast:,.0f}")
                
            except Exception as e:
                print(f"❌ ARIMAX prediction error: {e}")
                results["arimax_sales_forecast"] = 0.0
        
        # 6. KMEANS CLUSTERING
        if kmeans is not None:
            try:
                cluster_input = input_scaled[:, :2]  # Use first 2 features
                cluster = int(kmeans.predict(cluster_input)[0])
                results["business_cluster"] = cluster
                
                cluster_names = {
                    0: "Small Business",
                    1: "Medium Business", 
                    2: "Large Business",
                    3: "Enterprise"
                }
                cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
                
                print(f"\n🎯 Business Classification: {cluster_name} (Cluster {cluster})")
                
            except Exception as e:
                print(f"❌ KMeans prediction error: {e}")
                results["business_cluster"] = 0
        
        # 7. ENSEMBLE AVERAGE (if optimized model available)
        if "optimized_mlp" in results:
            ensemble_predictions = []
            for model_name in ["optimized_mlp", "legacy_mlp", "decision_tree", "random_forest"]:
                if model_name in results:
                    ensemble_predictions.append(results[model_name])
            
            if ensemble_predictions:
                # Calculate weighted average (give more weight to optimized model)
                weights = [0.4, 0.2, 0.2, 0.2][:len(ensemble_predictions)]
                weights = [w / sum(weights) for w in weights]  # Normalize weights
                
                ensemble_modal = sum(w * pred["modal"] for w, pred in zip(weights, ensemble_predictions))
                ensemble_profit = sum(w * pred["profit"] for w, pred in zip(weights, ensemble_predictions))
                ensemble_rugi = sum(w * pred["rugi"] for w, pred in zip(weights, ensemble_predictions))
                
                results["ensemble_average"] = {
                    "modal": ensemble_modal,
                    "profit": ensemble_profit,
                    "rugi": ensemble_rugi
                }
                
                print(f"\n🎯 Ensemble Average (Weighted):")
                print(f"  • Modal: Rp {ensemble_modal:,.0f}")
                print(f"  • Profit: Rp {ensemble_profit:,.0f}")
                print(f"  • Rugi: Rp {ensemble_rugi:,.0f}")
        
        print(f"\n✅ Prediction completed using {len(results)} models")
        return results
        
    except Exception as e:
        print(f"❌ General prediction error: {e}")
        # Return default values if everything fails
        return {
            "optimized_mlp": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "legacy_mlp": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "decision_tree": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "random_forest": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "arimax_sales_forecast": 0.0,
            "business_cluster": 0
        }

# ------------------ Test Function ------------------ #
if __name__ == "__main__":
    print("🧪 Testing Optimized Ensemble Prediction System")
    print("=" * 60)
    
    # Test with sample business scenario
    test_pemasukan = 30_000_000  # 30M income
    test_pengeluaran = 15_000_000  # 15M expenses  
    test_jam = 0.5  # 12 hours operation (normalized)
    
    result = predict_all_optimized(test_pemasukan, test_pengeluaran, test_jam)
    
    print(f"\n📋 COMPLETE PREDICTION RESULTS:")
    print("=" * 60)
    
    for model_name, prediction in result.items():
        print(f"\n🔹 {model_name.upper().replace('_', ' ')}:")
        if isinstance(prediction, dict):
            for key, value in prediction.items():
                print(f"    {key.capitalize()}: Rp {value:,.0f}")
        else:
            print(f"    Result: {prediction}")
