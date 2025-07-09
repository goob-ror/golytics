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

# ------------------- IMPORT FUNCTIONS ------------------- #
sys.path.append(MAPPING_DIR)
sys.path.append(PREDICT_DIR)

try:
    from mapping.questionMap import extract_entities
    from predict.predictAll_optimized import predict_all_optimized
    print("✅ Optimized prediction system loaded")
except ImportError as e:
    print(f"⚠️ Could not load optimized system: {e}")
    try:
        from predict.predictAll import predict_all
        predict_all_optimized = predict_all  # Fallback to legacy
        print("📋 Using legacy prediction system")
    except ImportError:
        print("❌ No prediction system available")
        sys.exit(1)

# ------------------- ENHANCED INTERFACE ------------------- #
def tampilkan_intro():
    print("\n" + "=" * 70)
    print("🚀  OPTIMIZED BUSINESS INTELLIGENCE AI ASSISTANT")
    print("=" * 70)
    print("🎯 Powered by Advanced Machine Learning Models")
    print("📊 Features:")
    print("   • Optimized Neural Network (R² = 0.58)")
    print("   • Ensemble Predictions (Multiple Models)")
    print("   • Business Constraint Compliance")
    print("   • Real-time Sales Forecasting")
    print("   • Business Classification")
    print("\n💡 Example Questions:")
    print("   - Berapa keuntungan saya hari ini?")
    print("   - Apakah saya rugi minggu ini?")
    print("   - Berapa modal saya bulan ini?")
    print("   - Prediksi penjualan minggu depan")
    print("   - Klasifikasi bisnis saya")
    print("\n📝 Commands:")
    print("   - 'exit' atau 'quit' untuk keluar")
    print("   - 'help' untuk bantuan")
    print("   - 'demo' untuk contoh prediksi")
    print("   - 'test' untuk test semua model")
    print("\n" + "-" * 70)

def show_help():
    print("\n📚 BANTUAN PENGGUNAAN")
    print("-" * 50)
    print("🎯 Format Input:")
    print("   Masukkan pertanyaan dalam bahasa natural tentang:")
    print("   • Modal (capital)")
    print("   • Profit (keuntungan)")
    print("   • Rugi (kerugian)")
    print("   • Penjualan (sales)")
    print("   • Klasifikasi bisnis")
    print("\n📊 Data yang Digunakan:")
    print("   • Pemasukan: Rp 30,000,000 (default)")
    print("   • Pengeluaran: Rp 15,000,000 (default)")
    print("   • Jam Operasi: 12 jam/hari (default)")
    print("\n🔧 Untuk mengubah data input, edit nilai di kode")

def run_demo():
    print("\n🎬 DEMO PREDIKSI BISNIS")
    print("-" * 50)
    
    scenarios = [
        {
            "name": "Bisnis Kecil",
            "pemasukan": 10_000_000,
            "pengeluaran": 7_000_000,
            "jam": 8/24
        },
        {
            "name": "Bisnis Menengah", 
            "pemasukan": 30_000_000,
            "pengeluaran": 15_000_000,
            "jam": 12/24
        },
        {
            "name": "Bisnis Besar",
            "pemasukan": 100_000_000,
            "pengeluaran": 60_000_000,
            "jam": 16/24
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Input: Pemasukan=Rp{scenario['pemasukan']:,}, Pengeluaran=Rp{scenario['pengeluaran']:,}")
        
        try:
            hasil = predict_all_optimized(
                scenario['pemasukan'], 
                scenario['pengeluaran'], 
                scenario['jam']
            )
            
            # Show only key results for demo
            if "optimized_mlp" in hasil:
                pred = hasil["optimized_mlp"]
                print(f"   Prediksi: Modal=Rp{pred['modal']:,.0f}, Profit=Rp{pred['profit']:,.0f}, Rugi=Rp{pred['rugi']:,.0f}")
            
            if "business_cluster" in hasil:
                cluster_names = {0: "Small", 1: "Medium", 2: "Large", 3: "Enterprise"}
                cluster_name = cluster_names.get(hasil["business_cluster"], "Unknown")
                print(f"   Klasifikasi: {cluster_name} Business")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def run_comprehensive_test():
    print("\n🧪 COMPREHENSIVE MODEL TEST")
    print("=" * 60)
    
    # Test data
    test_pemasukan = 25_000_000
    test_pengeluaran = 18_000_000
    test_jam = 10/24
    
    print(f"📊 Test Input:")
    print(f"   • Pemasukan: Rp {test_pemasukan:,}")
    print(f"   • Pengeluaran: Rp {test_pengeluaran:,}")
    print(f"   • Jam Operasi: {test_jam:.2f} (10 jam)")
    
    try:
        hasil = predict_all_optimized(test_pemasukan, test_pengeluaran, test_jam)
        
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 60)
        
        model_order = ["optimized_mlp", "legacy_mlp", "decision_tree", "random_forest", "ensemble_average"]
        
        for model_name in model_order:
            if model_name in hasil:
                prediction = hasil[model_name]
                print(f"\n🔹 {model_name.upper().replace('_', ' ')}:")
                if isinstance(prediction, dict):
                    for key, value in prediction.items():
                        print(f"    {key.capitalize()}: Rp {value:,.0f}")
        
        # Show additional results
        if "arimax_sales_forecast" in hasil:
            print(f"\n📈 Sales Forecast: Rp {hasil['arimax_sales_forecast']:,.0f}")
        
        if "business_cluster" in hasil:
            cluster_names = {0: "Small Business", 1: "Medium Business", 2: "Large Business", 3: "Enterprise"}
            cluster_name = cluster_names.get(hasil["business_cluster"], f"Cluster {hasil['business_cluster']}")
            print(f"🎯 Business Classification: {cluster_name}")
        
        # Performance comparison
        if "optimized_mlp" in hasil and "legacy_mlp" in hasil:
            opt_profit = hasil["optimized_mlp"]["profit"]
            leg_profit = hasil["legacy_mlp"]["profit"]
            improvement = ((opt_profit - leg_profit) / abs(leg_profit) * 100) if leg_profit != 0 else 0
            
            print(f"\n📊 MODEL COMPARISON:")
            print(f"   Optimized vs Legacy Profit Difference: {improvement:+.1f}%")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def process_user_question(user_input, pemasukan=30_000_000, pengeluaran=15_000_000, jam=12/24):
    """Process user question and return predictions"""
    
    try:
        # Extract entities from question
        waktu, target = extract_entities(user_input)
        
        print(f"\n🔍 Question Analysis:")
        print(f"   • Target: {target}")
        print(f"   • Time Period: {waktu}")
        
        # Make predictions
        print(f"\n🚀 Running Optimized Predictions...")
        hasil = predict_all_optimized(pemasukan, pengeluaran, jam)
        
        # Display results based on target
        print(f"\n📊 PREDICTION RESULTS:")
        print("-" * 50)
        
        # Show optimized model result first (priority)
        if "optimized_mlp" in hasil:
            pred = hasil["optimized_mlp"]
            print(f"\n🧠 OPTIMIZED AI MODEL:")
            print(f"   • Modal (Capital): Rp {pred['modal']:,.0f}")
            print(f"   • Profit (Keuntungan): Rp {pred['profit']:,.0f}")
            print(f"   • Rugi (Kerugian): Rp {pred['rugi']:,.0f}")
            
            # Business insights
            if pred['profit'] > 0:
                print(f"   💰 Status: PROFITABLE ✅")
            else:
                print(f"   📉 Status: LOSS ⚠️")
        
        # Show ensemble average if available
        if "ensemble_average" in hasil:
            pred = hasil["ensemble_average"]
            print(f"\n🎯 ENSEMBLE CONSENSUS:")
            print(f"   • Modal: Rp {pred['modal']:,.0f}")
            print(f"   • Profit: Rp {pred['profit']:,.0f}")
            print(f"   • Rugi: Rp {pred['rugi']:,.0f}")
        
        # Show additional insights
        if "arimax_sales_forecast" in hasil:
            forecast = hasil["arimax_sales_forecast"]
            print(f"\n📈 SALES FORECAST (7-day): Rp {forecast:,.0f}")
        
        if "business_cluster" in hasil:
            cluster_names = {0: "Small Business", 1: "Medium Business", 2: "Large Business", 3: "Enterprise"}
            cluster_name = cluster_names.get(hasil["business_cluster"], "Unknown")
            print(f"🏢 BUSINESS TYPE: {cluster_name}")
        
        return hasil
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        return None

# ------------------- MAIN LOOP ------------------- #
def main():
    tampilkan_intro()
    
    # Default business parameters (can be modified)
    default_pemasukan = 30_000_000   # 30M income
    default_pengeluaran = 15_000_000  # 15M expenses
    default_jam = 12/24              # 12 hours operation
    
    print(f"\n📊 Current Business Parameters:")
    print(f"   • Pemasukan: Rp {default_pemasukan:,}")
    print(f"   • Pengeluaran: Rp {default_pengeluaran:,}")
    print(f"   • Jam Operasi: {default_jam*24:.0f} jam/hari")
    
    while True:
        try:
            user_input = input("\n❓ Pertanyaan Anda: ").strip()
            
            if user_input.lower() in ["exit", "quit", "keluar"]:
                print("👋 Terima kasih telah menggunakan Optimized Business AI!")
                print("🚀 Model telah dioptimasi dengan R² = 0.58")
                break
            
            elif user_input.lower() == "help":
                show_help()
            
            elif user_input.lower() == "demo":
                run_demo()
            
            elif user_input.lower() == "test":
                run_comprehensive_test()
            
            elif user_input.strip() == "":
                print("⚠️ Silakan masukkan pertanyaan atau ketik 'help' untuk bantuan")
            
            else:
                # Process the question
                hasil = process_user_question(
                    user_input, 
                    default_pemasukan, 
                    default_pengeluaran, 
                    default_jam
                )
                
                if hasil is None:
                    print("⚠️ Gagal memproses pertanyaan. Coba lagi atau ketik 'help'")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program dihentikan oleh user. Sampai jumpa!")
            break
        except Exception as e:
            print(f"⚠️ Terjadi kesalahan: {e}")
            print("💡 Ketik 'help' untuk bantuan atau 'test' untuk menguji sistem")

if __name__ == "__main__":
    main()
