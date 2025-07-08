import os
import json
import torch
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

# ------------------ DEFINISI MODEL ------------------ #
class BisnisAssistantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# ------------------ SETUP BASE PATH ------------------ #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "result", "model", "assistV1.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "generate", "dataset", "numeric", "lanjutan", "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "generate", "dataset", "numeric", "lanjutan", "scaler_y.pkl")
DATA_DIR = os.path.join(BASE_DIR, "generate", "dataset", "numeric", "lanjutan")

# ------------------ LOAD MODEL DAN SCALER ------------------ #
def load_pipeline():
    # Load model
    model = BisnisAssistantModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load scaler
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Load SBERT
    sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    return model, scaler_x, scaler_y, sbert

# ------------------ EKSTRAK ENTITAS DARI TEXT ------------------ #
def extract_entities(text: str):
    text = text.lower()

    if "hari ini" in text or "sekarang" in text:
        waktu = "hari_ini"
    elif "kemarin" in text:
        waktu = "kemarin"
    elif "minggu ini" in text:
        waktu = "minggu_ini"
    elif "bulan ini" in text:
        waktu = "bulan_ini"
    elif "tahun ini" in text:
        waktu = "tahun_ini"
    else:
        waktu = "all"

    if "modal" in text:
        target = "modal"
    elif "rugi" in text or "kerugian" in text:
        target = "rugi"
    elif "untung" in text or "profit" in text or "laba" in text or "keuntungan" in text:
        target = "profit"
    else:
        target = "profit"

    return waktu, target

# ------------------ AMBIL RATA-RATA DATA TRANSAKSI ------------------ #
def ambil_data_by_waktu(waktu_target="hari_ini", data_dir=DATA_DIR):
    matched = []
    now = datetime.now()

    for file in os.listdir(data_dir):
        if not file.endswith(".json") or file == "normalization_stats.json":
            continue

        with open(os.path.join(data_dir, file)) as f:
            data = json.load(f)

        for item in data:
            try:
                waktu = datetime.fromisoformat(item["waktu"])
                if waktu_target == "hari_ini" and waktu.date() == now.date():
                    matched.append(item)
                elif waktu_target == "kemarin" and waktu.date() == (now.date() - timedelta(days=1)):
                    matched.append(item)
                elif waktu_target == "minggu_ini" and waktu.isocalendar()[1] == now.isocalendar()[1]:
                    matched.append(item)
                elif waktu_target == "bulan_ini" and waktu.month == now.month and waktu.year == now.year:
                    matched.append(item)
                elif waktu_target == "tahun_ini" and waktu.year == now.year:
                    matched.append(item)
                elif waktu_target == "all":
                    matched.append(item)
            except:
                continue

    if not matched:
        raise ValueError(f"Tidak ada data yang cocok untuk waktu: {waktu_target}")

    pemasukan = np.mean([item["total_pemasukan"] for item in matched])
    pengeluaran = np.mean([item["total_pengeluaran"] for item in matched])
    jam = np.mean([datetime.fromisoformat(item["waktu"]).hour / 24.0 for item in matched])

    return pemasukan, pengeluaran, jam

# ------------------ PREDIKSI ------------------ #
def predict(model, scaler_x, scaler_y, pemasukan, pengeluaran, jam):
    input_data = np.array([[pemasukan, pengeluaran, jam]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)

    with torch.no_grad():
        prediction_scaled = model(torch.tensor(input_scaled)).numpy()

    prediction = scaler_y.inverse_transform(prediction_scaled)[0]

    return {
        "modal": prediction[0],
        "profit": prediction[1],
        "rugi": prediction[2]
    }
