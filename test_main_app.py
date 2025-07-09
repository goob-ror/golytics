#!/usr/bin/env python3
"""
Test the main application with a single question
"""

import sys
import os
sys.path.append('ensemble')

from ensemble.predict.predictAll import predict_all
from ensemble.mapping.questionMap import extract_entities

def simulate_main_app():
    """Simulate the main app with a test question"""
    print("=======================================")
    print("🤖  BUSINESS INTELLIGENCE AI ASSISTANT  ")
    print("=======================================")
    print()
    print("Contoh pertanyaan:")
    print("- Berapa keuntungan saya hari ini?     ")
    print("- Apakah saya rugi minggu ini?")
    print("- Berapa modal saya bulan ini?")
    print("- Bandingkan toko A dan toko B")
    print()
    
    # Simulate user input
    user_input = "Berapa keuntungan saya hari ini?"
    print(f"❓ Pertanyaan Anda: {user_input}")
    
    try:
        waktu, target = extract_entities(user_input)
        
        print(f"\n🧠 Interpretasi pertanyaan:")
        print(f"- Target yang diminta: {target}")
        print(f"- Rentang waktu: {waktu}")
        
        # Use realistic business data
        pemasukan = 30000000   # 30M income
        pengeluaran = 15000000 # 15M expenses  
        jam = 12 / 24         # 12 hours operation
        
        print("\n🔮 Melakukan prediksi semua model...")
        print()
        
        hasil = predict_all(pemasukan, pengeluaran, jam)
        
        for model, val in hasil.items():
            print(f"[{model.upper()}]")
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"  {k.capitalize()}: {v:.2f}")
            else:
                print(f"  Hasil: {val}")
            print("")
            
        print("✅ Prediksi berhasil! Semua model berfungsi dengan baik.")
        
    except Exception as e:
        print(f"⚠️  Terjadi kesalahan: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_main_app()
