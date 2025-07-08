import sys
import os

# Tambahkan path ke folder utama (tempat pipelineMlp.py berada)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelineMlp import load_pipeline, extract_entities, ambil_data_by_waktu, predict


# ------------------- MAIN INTERFACE ------------------- #
if __name__ == "__main__":
    print("� Business Tracker AI Siap!")
    print("Tanyakan sesuatu seperti:")
    print("- Berapa keuntungan saya hari ini?")
    print("- Apakah saya rugi minggu ini?")
    print("- Berapa modal saya bulan ini?\n")
    print("Ketik 'exit' untuk keluar.\n")

    # Load pipeline
    try:
        model, scaler_x, scaler_y, sbert = load_pipeline()
    except Exception as e:
        print(f"❌ Gagal memuat pipeline: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("�️ Anda: ").strip()
            if user_input.lower() in ["exit", "keluar", "quit"]:
                print("� Sampai jumpa!")
                break

            # Ekstraksi entitas dari pertanyaan
            waktu, target = extract_entities(user_input)

            # Ambil data transaksi berdasarkan waktu
            pemasukan, pengeluaran, jam = ambil_data_by_waktu(waktu)

            # Prediksi hasil
            hasil = predict(model, scaler_x, scaler_y, pemasukan, pengeluaran, jam)

            # Tampilkan hasil
            print(f"\n� Prediksi untuk waktu: {waktu.replace('_', ' ')}")
            print(f"→ {target.capitalize()}: Rp {hasil[target]:,.0f}\n")

        except Exception as e:
            print(f"⚠️ Terjadi kesalahan: {e}\n")
