#!/usr/bin/env python3
"""
Test Updated Ensemble System
Verifies that all updates are working correctly
"""

import sys
import os

def test_optimized_prediction():
    """Test the optimized prediction system"""
    print("🧪 Testing Optimized Prediction System")
    print("-" * 50)
    
    try:
        sys.path.append('ensemble/predict')
        from predictAll_optimized import predict_all_optimized
        
        # Test with sample data
        pemasukan = 30_000_000
        pengeluaran = 15_000_000
        jam = 0.5
        
        print(f"📊 Test Input:")
        print(f"   • Pemasukan: Rp {pemasukan:,}")
        print(f"   • Pengeluaran: Rp {pengeluaran:,}")
        print(f"   • Jam: {jam}")
        
        result = predict_all_optimized(pemasukan, pengeluaran, jam)
        
        print(f"\n✅ Optimized System Working!")
        print(f"📋 Available Models: {list(result.keys())}")
        
        # Show key results
        if "optimized_mlp" in result:
            pred = result["optimized_mlp"]
            print(f"\n🧠 Optimized MLP:")
            print(f"   • Modal: Rp {pred['modal']:,.0f}")
            print(f"   • Profit: Rp {pred['profit']:,.0f}")
            print(f"   • Rugi: Rp {pred['rugi']:,.0f}")
        
        if "ensemble_average" in result:
            pred = result["ensemble_average"]
            print(f"\n🎯 Ensemble Average:")
            print(f"   • Modal: Rp {pred['modal']:,.0f}")
            print(f"   • Profit: Rp {pred['profit']:,.0f}")
            print(f"   • Rugi: Rp {pred['rugi']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized system failed: {e}")
        return False

def test_legacy_prediction():
    """Test the legacy prediction system"""
    print("\n🧪 Testing Legacy Prediction System")
    print("-" * 50)
    
    try:
        sys.path.append('ensemble/predict')
        from predictAll import predict_all
        
        # Test with sample data
        pemasukan = 30_000_000
        pengeluaran = 15_000_000
        jam = 0.5
        
        result = predict_all(pemasukan, pengeluaran, jam)
        
        print(f"✅ Legacy System Working!")
        print(f"📋 Available Models: {list(result.keys())}")
        
        # Show key results
        if "mlp" in result:
            pred = result["mlp"]
            print(f"\n🧠 MLP:")
            print(f"   • Modal: Rp {pred['modal']:,.0f}")
            print(f"   • Profit: Rp {pred['profit']:,.0f}")
            print(f"   • Rugi: Rp {pred['rugi']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy system failed: {e}")
        return False

def test_pipeline():
    """Test the updated pipeline"""
    print("\n🧪 Testing Updated Pipeline")
    print("-" * 50)
    
    try:
        sys.path.append('ensemble/pipeline')
        from golyticsPipeline import run_prediction
        
        # Test with sample data
        pemasukan = 25_000_000
        pengeluaran = 18_000_000
        jam = 0.4
        
        print(f"📊 Pipeline Test Input:")
        print(f"   • Pemasukan: Rp {pemasukan:,}")
        print(f"   • Pengeluaran: Rp {pengeluaran:,}")
        print(f"   • Jam: {jam}")
        
        result = run_prediction(pemasukan, pengeluaran, jam)
        
        print(f"\n✅ Pipeline Working!")
        print(f"📋 Pipeline Results:")
        
        for model_name, prediction in result.items():
            print(f"\n🔹 {model_name.upper()}:")
            if isinstance(prediction, dict):
                for key, value in prediction.items():
                    print(f"    {key.capitalize()}: Rp {value:,.0f}")
            else:
                print(f"    Result: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def test_main_interface():
    """Test the main interface imports"""
    print("\n🧪 Testing Main Interface")
    print("-" * 50)
    
    try:
        sys.path.append('ensemble')
        sys.path.append('ensemble/mapping')
        
        # Test imports
        from mapping.questionMap import extract_entities
        
        # Test question mapping
        test_question = "Berapa keuntungan saya hari ini?"
        waktu, target = extract_entities(test_question)
        
        print(f"✅ Question Mapping Working!")
        print(f"📝 Test Question: '{test_question}'")
        print(f"   • Target: {target}")
        print(f"   • Time: {waktu}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main interface failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING UPDATED ENSEMBLE SYSTEM")
    print("=" * 70)
    print("🎯 Verifying all components are working correctly")
    print("=" * 70)
    
    tests = [
        ("Optimized Prediction", test_optimized_prediction),
        ("Legacy Prediction", test_legacy_prediction),
        ("Updated Pipeline", test_pipeline),
        ("Main Interface", test_main_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n🏆 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ensemble system is ready for use.")
        print("\n🚀 You can now run:")
        print("   • python ensemble/main.py - Interactive AI assistant")
        print("   • python demo_optimized_ensemble.py - Comprehensive demo")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All systems operational!")
    else:
        print("\n❌ Some issues detected!")
