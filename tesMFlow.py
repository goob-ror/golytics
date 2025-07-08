import os
import json
import torch
import joblib
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from train import BisnisAssistantModel  # class model diambil dari train.py

# ------------------- LOAD MODEL NEURAL NETWORK ------------------- #
model = BisnisAssistantModel()
model.load_state_dict(torch.load("result/model/assistV1.pth", map_location=torch.device('cpu')))
model.eval()

scaler_x = joblib.load("generate/dataset/numeric/lanjutan/scaler_x.pkl")
scaler_y = joblib.load("generate/dataset/numeric/lanjutan/scaler_y.pkl")

# ------------------- LOAD MODEL SBERT ------------------- #
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# ------------------- EKSTRAK ENTITAS DARI INPUT USER ------------------- #
def extract_entities_dari_pertanyaan(text):
    text = text.lower()

    # Deteksi waktu
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
        waktu = "all"  # fallback

    # Deteksi target
    if "modal" in text:
        target = "modal"
    elif "rugi" in text or "kerugian" in text:
        target = "rugi"
    elif "untung" in text or "profit" in text or "laba" in text or "keuntungan" in text:
        target = "profit"
    else:
        target = "profit"  # default target

    return waktu, target

# ------------------- AMBIL RATA-RATA DATA BERDASARKAN WAKTU ------------------- #
def ambil_data_by_waktu(waktu_target="hari_ini"):
    matched = []
    data_dir = "generate/dataset/numeric/lanjutan"
    now = datetime.now()

    for file in os.listdir(data_dir):
        if not file.endswith(".json") or "normalization" in file:
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
            except Exception as e:
                continue

    if not matched:
        raise ValueError(f"Tidak ada data untuk waktu: {waktu_target}")

    pemasukan = np.mean([item["total_pemasukan"] for item in matched])
    pengeluaran = np.mean([item["total_pengeluaran"] for item in matched])
    jam = np.mean([datetime.fromisoformat(item["waktu"]).hour / 24.0 for item in matched])

    return pemasukan, pengeluaran, jam

# ------------------- FUNGSI PREDIKSI ------------------- #
def predict_from_data(pemasukan, pengeluaran, jam_float):
    input_data = np.array([[pemasukan, pengeluaran, jam_float]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)

    with torch.no_grad():
        output_scaled = model(torch.tensor(input_scaled)).numpy()

    output = scaler_y.inverse_transform(output_scaled)[0]
    return {
        "modal": output[0],
        "profit": output[1],
        "rugi": output[2]
    }

# ------------------- MAIN INTERFACE ------------------- #
if __name__ == "__main__":
    print("� Business Tracker AI Siap!")
    print("Tanyakan sesuatu seperti:")
    print("- Berapa keuntungan saya hari ini?")
    print("- Apakah saya rugi minggu ini?")
    print("- Berapa modal saya bulan ini?\n")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            user_input = input("� Anda: ").strip()
            if user_input.lower() in ["exit", "keluar", "quit"]:
                print("� Sampai jumpa!")
                break

            waktu, target = extract_entities_dari_pertanyaan(user_input)
            pemasukan, pengeluaran, jam = ambil_data_by_waktu(waktu)
            hasil = predict_from_data(pemasukan, pengeluaran, jam)

            print(f"� Prediksi untuk waktu: {waktu.replace('_', ' ')}")
            print(f"→ {target.capitalize()}: Rp {hasil[target]:,.0f}\n")

        except Exception as e:
            print(f"⚠️ Terjadi kesalahan: {e}\n")
