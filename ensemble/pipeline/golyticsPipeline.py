import os
import sys
import numpy as np
import torch
import joblib

# Setup relative path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

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
    mlp_model, tree, rf, arimax, kmeans, scaler_x, scaler_y = load_all_models()

    # Input 7 fitur dummy (disamakan dengan preprocessing)
    input_data = np.array([[1, 1, 1, 1, 0, 1, 1]], dtype=np.float32)
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
        "mlp": {"modal": pred_mlp[0], "profit": pred_mlp[1], "rugi": pred_mlp[2]},
        "tree": {"modal": pred_tree[0], "profit": pred_tree[1], "rugi": pred_tree[2]},
        "rf": {"modal": pred_rf[0], "profit": pred_rf[1], "rugi": pred_rf[2]},
        "arimax_sales": pred_arimax,
        "kmeans_cluster": cluster
    }
