import sys
import os
import torch
import joblib
import numpy as np

# ------------------- PATH SETUP ------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MAPPING_DIR = os.path.join(BASE_DIR, "mapping")
PREDICT_DIR = os.path.join(BASE_DIR, "predict")

# ------------------- IMPORT FUNGSI ------------------- #
sys.path.append(MAPPING_DIR)
sys.path.append(PREDICT_DIR)

from mapping.questionMap import extract_entities  # Dari mapping/questionMap.py
from predict.predictAll import predict_all        # Dari predict/predictAll.py

# ------------------- INTERFACE ------------------- #
def tampilkan_intro():
    print("\n=======================================")
    print("�  BUSINESS INTELLIGENCE AI ASSISTANT ")
    print("=======================================\n")
    print("Contoh pertanyaan:")
    print("- Berapa keuntungan saya hari ini?")
    print("- Apakah saya rugi minggu ini?")
    print("- Berapa modal saya bulan ini?")
    print("- Bandingkan toko A dan toko B\n")
    print("Ketik 'exit' untuk keluar.\n")

# ------------------- MAIN LOOP ------------------- #
if __name__ == "__main__":
    tampilkan_intro()

    while True:
        try:
            user_input = input("❓ Pertanyaan Anda: ").strip()
            if user_input.lower() == "exit":
                print("� Terima kasih. Sampai jumpa!")
                break

            waktu, target = extract_entities(user_input)

            print(f"\n� Interpretasi pertanyaan:")
            print(f"- Target yang diminta: {target}")
            print(f"- Rentang waktu: {waktu}")

            # Dummy data input (ganti sesuai kebutuhan)
            pemasukan = 30000000
            pengeluaran = 15000000
            jam = 12 / 24

            print("\n� Melakukan prediksi semua model...\n")
            hasil = predict_all(pemasukan, pengeluaran, jam)

            for model, val in hasil.items():
                print(f"[{model.upper()}]")
                if isinstance(val, dict):
                    for k, v in val.items():
                        print(f"  {k.capitalize()}: {v:.2f}")
                else:
                    print(f"  Hasil: {val}")
                print("")

        except Exception as e:
            print(f"⚠️  Terjadi kesalahan: {e}\n")
