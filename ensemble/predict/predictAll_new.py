import torch
import joblib
import numpy as np
import os
import torch.nn as nn

# ------------------ Model MLP ------------------ #
class BisnisAssistantModel(nn.Module):
    """MLP Model for business prediction - matches training architecture"""
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
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
        return self.model(x)

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
        # MLP Model
        mlp_model = BisnisAssistantModel(input_size=3, hidden_size=64, output_size=3)
        mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_model.pth"), 
                                           map_location='cpu'))
        mlp_model.eval()

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
        
        # MLP Prediction
        try:
            with torch.no_grad():
                pred_mlp = mlp_model(torch.tensor(input_scaled, dtype=torch.float32)).numpy()
                pred_mlp = scaler_y.inverse_transform(pred_mlp)[0]
                results["mlp"] = {
                    "modal": float(pred_mlp[0]), 
                    "profit": float(pred_mlp[1]), 
                    "rugi": float(pred_mlp[2])
                }
        except Exception as e:
            print(f"MLP prediction error: {e}")
            results["mlp"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}

        # Tree Prediction
        try:
            pred_tree = tree.predict(input_scaled)
            pred_tree = scaler_y.inverse_transform(pred_tree.reshape(1, -1))[0]
            results["tree"] = {
                "modal": float(pred_tree[0]), 
                "profit": float(pred_tree[1]), 
                "rugi": float(pred_tree[2])
            }
        except Exception as e:
            print(f"Tree prediction error: {e}")
            results["tree"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}

        # Random Forest Prediction
        try:
            pred_rf = rf.predict(input_scaled)
            pred_rf = scaler_y.inverse_transform(pred_rf.reshape(1, -1))[0]
            results["rf"] = {
                "modal": float(pred_rf[0]), 
                "profit": float(pred_rf[1]), 
                "rugi": float(pred_rf[2])
            }
        except Exception as e:
            print(f"RF prediction error: {e}")
            results["rf"] = {"modal": 0.0, "profit": 0.0, "rugi": 0.0}

        # ARIMAX Prediction
        try:
            if arimax is not None:
                # Use dummy exogenous variables
                exog_input = np.array([[1, 0]])  # Promo=1, SchoolHoliday=0
                pred_arimax = arimax.forecast(steps=1, exog=exog_input)
                if hasattr(pred_arimax, 'iloc'):
                    pred_arimax = float(pred_arimax.iloc[0])
                else:
                    pred_arimax = float(pred_arimax[0])
                results["arimax_sales"] = pred_arimax
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
