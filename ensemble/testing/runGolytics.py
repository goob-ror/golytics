import sys
import os

# Tambahkan path ke ensemble/pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
sys.path.append(PIPELINE_DIR)

from pipeline import run_prediction

def test_run_prediction():
    # Dummy input
    pemasukan = 30000000
    pengeluaran = 15000000
    jam = 12 / 24

    print("✅ Menjalankan tes prediksi...\n")
    result = run_prediction(pemasukan, pengeluaran, jam)

    # Tampilkan hasil
    for model, output in result.items():
        print(f"� Model: {model}")
        if isinstance(output, dict):
            for key, val in output.items():
                print(f"  {key}: {val:.2f}")
        else:
            print(f"  Hasil: {output}")
        print()

    # Tes sederhana validasi struktur
    assert "mlp" in result and "profit" in result["mlp"]
    assert isinstance(result["mlp"]["modal"], float)
    print("✅ Test selesai tanpa error.\n")

if __name__ == "__main__":
    test_run_prediction()
