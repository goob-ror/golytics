import os
import sys
import numpy as np
import torch
import joblib

# Setup relative path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREDICT_DIR = os.path.join(BASE_DIR, "predict")

# Add predict directory to path for imports
sys.path.append(PREDICT_DIR)

# Try to import optimized prediction system
try:
    from predictAll_optimized import predict_all_optimized as predict_all
    USING_OPTIMIZED = True
    print("✅ Pipeline using Optimized Prediction System (R² = 0.58)")
except ImportError:
    print("⚠️ Optimized system not available, using legacy pipeline")
    USING_OPTIMIZED = False

# ------------------ Model Neural Network ------------------ #
class BisnisAssistantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(7, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# ------------------ Load Semua Komponen Model ------------------ #
def load_all_models():
    # Load scaler
    scaler_x = joblib.load(os.path.join(DATA_DIR, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(DATA_DIR, "scaler_y.pkl"))

    # Load MLP
    mlp_model = BisnisAssistantModel()
    mlp_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp_model.pth")))
    mlp_model.eval()

    # Load lainnya
    tree = joblib.load(os.path.join(MODEL_DIR, "tree_model.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    arimax = joblib.load(os.path.join(MODEL_DIR, "arimax_model.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

    return mlp_model, tree, rf, arimax, kmeans, scaler_x, scaler_y

# ------------------ Prediksi Lengkap ------------------ #
def run_prediction(pemasukan, pengeluaran, jam):
    """
    Enhanced prediction function that uses optimized system if available

    Args:
        pemasukan (float): Income/revenue
        pengeluaran (float): Expenses
        jam (float): Operating hours (0-1 normalized)

    Returns:
        dict: Predictions from all available models
    """

    if USING_OPTIMIZED:
        # Use optimized prediction system
        try:
            print(f"🚀 Using optimized pipeline for prediction...")
            results = predict_all(pemasukan, pengeluaran, jam)

            # Convert to legacy format for compatibility
            legacy_format = {}

            # Map optimized results to legacy format
            if "optimized_mlp" in results:
                legacy_format["mlp"] = results["optimized_mlp"]
            elif "legacy_mlp" in results:
                legacy_format["mlp"] = results["legacy_mlp"]

            if "decision_tree" in results:
                legacy_format["tree"] = results["decision_tree"]

            if "random_forest" in results:
                legacy_format["rf"] = results["random_forest"]

            if "arimax_sales_forecast" in results:
                legacy_format["arimax_sales"] = results["arimax_sales_forecast"]

            if "business_cluster" in results:
                legacy_format["kmeans_cluster"] = results["business_cluster"]

            # Add ensemble result if available
            if "ensemble_average" in results:
                legacy_format["ensemble"] = results["ensemble_average"]

            return legacy_format

        except Exception as e:
            print(f"⚠️ Optimized pipeline failed: {e}")
            print("📋 Falling back to legacy pipeline...")

    # Legacy pipeline implementation
    try:
        mlp_model, tree, rf, arimax, kmeans, scaler_x, scaler_y = load_all_models()

        # Input 3 features (updated to match optimized system)
        input_data = np.array([[pemasukan, pengeluaran, jam]], dtype=np.float32)
        input_scaled = scaler_x.transform(input_data)

        # Prediksi MLP
        pred_mlp = mlp_model(torch.tensor(input_scaled, dtype=torch.float32)).detach().numpy()
        pred_mlp = scaler_y.inverse_transform(pred_mlp)[0]

        # Tree & RF
        pred_tree = scaler_y.inverse_transform(tree.predict(input_scaled).reshape(1, -1))[0]
        pred_rf = scaler_y.inverse_transform(rf.predict(input_scaled).reshape(1, -1))[0]

        # ARIMAX (menggunakan dummy promo + school holiday)
        arimax_input = np.array([[1, 0]])
        pred_arimax = arimax.forecast(steps=1, exog=arimax_input)[0]

        # KMeans cluster
        cluster = int(kmeans.predict(input_data)[0])

        return {
            "mlp": {"modal": float(pred_mlp[0]), "profit": float(pred_mlp[1]), "rugi": float(pred_mlp[2])},
            "tree": {"modal": float(pred_tree[0]), "profit": float(pred_tree[1]), "rugi": float(pred_tree[2])},
            "rf": {"modal": float(pred_rf[0]), "profit": float(pred_rf[1]), "rugi": float(pred_rf[2])},
            "arimax_sales": float(pred_arimax),
            "kmeans_cluster": cluster
        }

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        # Return default values
        return {
            "mlp": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "tree": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "rf": {"modal": 0.0, "profit": 0.0, "rugi": 0.0},
            "arimax_sales": 0.0,
            "kmeans_cluster": 0
        }
