import torch
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ------------------ Model MLP ------------------ #
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

# ------------------ Fungsi Prediksi Utama ------------------ #
def predict_all(pemasukan, pengeluaran, jam):
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "../data")
    model_path = os.path.join(base_path, "../models")

    # ------------------ Load Scaler ------------------ #
    scaler_x = joblib.load(os.path.join(data_path, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(data_path, "scaler_y.pkl"))

    # ------------------ Siapkan Input Dinamis ------------------ #
    profit = pemasukan - pengeluaran
    rugi = max(pengeluaran - pemasukan, 0)
    libur = 0  # Dummy: anggap tidak libur
    hari = datetime.today().weekday()  # 0=Senin, 6=Minggu

    # Format: [pemasukan, pengeluaran, profit, rugi, libur, jam_operasional, hari]
    input_data = np.array([[pemasukan, pengeluaran, profit, rugi, libur, jam, hari]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)

    # ------------------ Load Semua Model ------------------ #
    mlp_model = BisnisAssistantModel()
    mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_model.pth")))
    mlp_model.eval()

    tree = joblib.load(os.path.join(model_path, "tree_model.pkl"))
    rf = joblib.load(os.path.join(model_path, "rf_model.pkl"))
    arimax = joblib.load(os.path.join(model_path, "arimax_model.pkl"))
    kmeans = joblib.load(os.path.join(model_path, "kmeans_model.pkl"))

    # ------------------ Prediksi Model ------------------ #
    # MLP
    pred_mlp = mlp_model(torch.tensor(input_scaled, dtype=torch.float32)).detach().numpy()
    pred_mlp = scaler_y.inverse_transform(pred_mlp)[0]

    # Tree
    pred_tree = scaler_y.inverse_transform(tree.predict(input_scaled).reshape(1, -1))[0]

    # Random Forest
    pred_rf = scaler_y.inverse_transform(rf.predict(input_scaled).reshape(1, -1))[0]

    # ARIMAX
    arimax_input = np.array([[1, 0]])  # Promo=1, SchoolHoliday=0
    pred_arimax = arimax.forecast(steps=1, exog=arimax_input)

    if isinstance(pred_arimax, pd.Series):
        pred_arimax = pred_arimax.iloc[0]
    else:
        pred_arimax = float(pred_arimax[0])

    # KMeans (gunakan fitur 0,1,2,4,6)
    input_kmeans = input_data[:, [0, 1, 2, 4, 6]]
    input_kmeans = np.array(input_kmeans, dtype=np.float64, order="C")
    cluster = int(kmeans.predict(input_kmeans)[0])

    # ------------------ Return Semua ------------------ #
    return {
        "mlp": {"modal": pred_mlp[0], "profit": pred_mlp[1], "rugi": pred_mlp[2]},
        "tree": {"modal": pred_tree[0], "profit": pred_tree[1], "rugi": pred_tree[2]},
        "rf": {"modal": pred_rf[0], "profit": pred_rf[1], "rugi": pred_rf[2]},
        "arimax_sales": pred_arimax,
        "kmeans_cluster": cluster
    }

# ------------------ Tes Manual ------------------ #
if __name__ == "__main__":
    result = predict_all(pemasukan=30000000, pengeluaran=15000000, jam=12 / 24)
    print("Hasil prediksi semua model:\n")
    for model, val in result.items():
        print(f"{model.upper()}:")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k.capitalize()}: {v:.2f}")
        else:
            print(f"  Hasil: {val}")
        print()
