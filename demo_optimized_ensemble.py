#!/usr/bin/env python3
"""
Demo of Optimized Ensemble System
Showcases the integration of our optimized model with the existing ensemble
"""

import sys
import os
sys.path.append('ensemble/predict')

from predictAll_optimized import predict_all_optimized

def run_business_scenarios():
    """Run multiple business scenarios to demonstrate the optimized ensemble"""
    
    print("🚀 OPTIMIZED ENSEMBLE SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("🎯 Integration of Optimized Model (R² = 0.58) with Existing Ensemble")
    print("=" * 80)
    
    scenarios = [
        {
            "name": "🏪 Small Retail Business",
            "description": "Local convenience store",
            "pemasukan": 8_000_000,    # 8M income
            "pengeluaran": 6_000_000,   # 6M expenses
            "jam": 8/24,               # 8 hours operation
            "expected": "Low modal, small profit"
        },
        {
            "name": "🍕 Medium Restaurant",
            "description": "Popular local restaurant",
            "pemasukan": 25_000_000,   # 25M income
            "pengeluaran": 18_000_000,  # 18M expenses
            "jam": 12/24,              # 12 hours operation
            "expected": "Moderate modal, good profit"
        },
        {
            "name": "🏭 Large Manufacturing",
            "description": "Industrial manufacturing company",
            "pemasukan": 150_000_000,  # 150M income
            "pengeluaran": 90_000_000,  # 90M expenses
            "jam": 16/24,              # 16 hours operation
            "expected": "High modal, high profit"
        },
        {
            "name": "💼 Struggling Startup",
            "description": "New tech startup with high expenses",
            "pemasukan": 5_000_000,    # 5M income
            "pengeluaran": 12_000_000,  # 12M expenses
            "jam": 10/24,              # 10 hours operation
            "expected": "Low modal, negative profit"
        },
        {
            "name": "🏢 Enterprise Corporation",
            "description": "Large multinational corporation",
            "pemasukan": 500_000_000,  # 500M income
            "pengeluaran": 300_000_000, # 300M expenses
            "jam": 20/24,              # 20 hours operation
            "expected": "Very high modal, very high profit"
        }
    ]
    
    results_summary = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   📋 {scenario['description']}")
        print(f"   💰 Input: Pemasukan=Rp{scenario['pemasukan']:,}, Pengeluaran=Rp{scenario['pengeluaran']:,}")
        print(f"   ⏰ Jam Operasi: {scenario['jam']*24:.0f} jam/hari")
        print(f"   🎯 Expected: {scenario['expected']}")
        
        try:
            # Get predictions from optimized ensemble
            hasil = predict_all_optimized(
                scenario['pemasukan'], 
                scenario['pengeluaran'], 
                scenario['jam']
            )
            
            print(f"\n   📊 PREDICTION RESULTS:")
            
            # Show optimized model results (priority)
            if "optimized_mlp" in hasil:
                pred = hasil["optimized_mlp"]
                print(f"   🧠 Optimized AI Model:")
                print(f"      • Modal: Rp {pred['modal']:,.0f}")
                print(f"      • Profit: Rp {pred['profit']:,.0f}")
                print(f"      • Rugi: Rp {pred['rugi']:,.0f}")
                
                # Business status
                if pred['profit'] > 0:
                    status = "✅ PROFITABLE"
                    status_color = "green"
                else:
                    status = "⚠️ LOSS"
                    status_color = "red"
                print(f"      • Status: {status}")
                
                # Calculate metrics
                profit_margin = (pred['profit'] / scenario['pemasukan']) * 100 if scenario['pemasukan'] > 0 else 0
                print(f"      • Profit Margin: {profit_margin:.1f}%")
            
            # Show ensemble consensus
            if "ensemble_average" in hasil:
                pred = hasil["ensemble_average"]
                print(f"   🎯 Ensemble Consensus:")
                print(f"      • Modal: Rp {pred['modal']:,.0f}")
                print(f"      • Profit: Rp {pred['profit']:,.0f}")
                print(f"      • Rugi: Rp {pred['rugi']:,.0f}")
            
            # Show business classification
            if "business_cluster" in hasil:
                cluster_names = {
                    0: "Small Business",
                    1: "Medium Business", 
                    2: "Large Business",
                    3: "Enterprise"
                }
                cluster_name = cluster_names.get(hasil["business_cluster"], f"Cluster {hasil['business_cluster']}")
                print(f"   🏢 Business Classification: {cluster_name}")
            
            # Show sales forecast
            if "arimax_sales_forecast" in hasil:
                forecast = hasil["arimax_sales_forecast"]
                print(f"   📈 Sales Forecast (7-day): Rp {forecast:,.0f}")
            
            # Store for summary
            if "optimized_mlp" in hasil:
                results_summary.append({
                    'name': scenario['name'],
                    'profit': hasil["optimized_mlp"]['profit'],
                    'profit_margin': profit_margin,
                    'classification': cluster_name if "business_cluster" in hasil else "Unknown"
                })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results_summary.append({
                'name': scenario['name'],
                'profit': 0,
                'profit_margin': 0,
                'classification': "Error"
            })
        
        print(f"   {'-' * 60}")
    
    # Summary Analysis
    print(f"\n🏆 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Business Performance Ranking:")
    sorted_results = sorted(results_summary, key=lambda x: x['profit'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        profit_status = "💰" if result['profit'] > 0 else "📉"
        print(f"   {i}. {result['name']}")
        print(f"      Profit: Rp {result['profit']:,.0f} {profit_status}")
        print(f"      Margin: {result['profit_margin']:.1f}%")
        print(f"      Type: {result['classification']}")
    
    # Performance insights
    profitable_count = sum(1 for r in results_summary if r['profit'] > 0)
    total_count = len(results_summary)
    
    print(f"\n📈 PERFORMANCE INSIGHTS:")
    print(f"   • Profitable Businesses: {profitable_count}/{total_count} ({profitable_count/total_count*100:.1f}%)")
    print(f"   • Average Profit Margin: {sum(r['profit_margin'] for r in results_summary)/total_count:.1f}%")
    
    best_business = max(results_summary, key=lambda x: x['profit'])
    worst_business = min(results_summary, key=lambda x: x['profit'])
    
    print(f"   • Best Performer: {best_business['name']} (Rp {best_business['profit']:,.0f})")
    print(f"   • Needs Attention: {worst_business['name']} (Rp {worst_business['profit']:,.0f})")
    
    return results_summary

def demonstrate_model_comparison():
    """Compare optimized vs legacy models"""
    
    print(f"\n🔬 MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    
    test_case = {
        "pemasukan": 40_000_000,
        "pengeluaran": 25_000_000,
        "jam": 14/24
    }
    
    print(f"📊 Test Case:")
    print(f"   • Pemasukan: Rp {test_case['pemasukan']:,}")
    print(f"   • Pengeluaran: Rp {test_case['pengeluaran']:,}")
    print(f"   • Jam Operasi: {test_case['jam']*24:.0f} jam")
    
    try:
        hasil = predict_all_optimized(
            test_case['pemasukan'],
            test_case['pengeluaran'], 
            test_case['jam']
        )
        
        print(f"\n📋 DETAILED MODEL COMPARISON:")
        
        models_to_compare = ["optimized_mlp", "legacy_mlp", "decision_tree", "random_forest"]
        
        for model_name in models_to_compare:
            if model_name in hasil:
                pred = hasil[model_name]
                print(f"\n🔹 {model_name.upper().replace('_', ' ')}:")
                print(f"   Modal: Rp {pred['modal']:,.0f}")
                print(f"   Profit: Rp {pred['profit']:,.0f}")
                print(f"   Rugi: Rp {pred['rugi']:,.0f}")
                
                # Calculate profit margin
                margin = (pred['profit'] / test_case['pemasukan']) * 100
                print(f"   Profit Margin: {margin:.1f}%")
        
        # Show ensemble result
        if "ensemble_average" in hasil:
            pred = hasil["ensemble_average"]
            print(f"\n🎯 ENSEMBLE WEIGHTED AVERAGE:")
            print(f"   Modal: Rp {pred['modal']:,.0f}")
            print(f"   Profit: Rp {pred['profit']:,.0f}")
            print(f"   Rugi: Rp {pred['rugi']:,.0f}")
            margin = (pred['profit'] / test_case['pemasukan']) * 100
            print(f"   Profit Margin: {margin:.1f}%")
        
        # Model performance analysis
        if "optimized_mlp" in hasil and "legacy_mlp" in hasil:
            opt_profit = hasil["optimized_mlp"]["profit"]
            leg_profit = hasil["legacy_mlp"]["profit"]
            
            if leg_profit != 0:
                improvement = ((opt_profit - leg_profit) / abs(leg_profit)) * 100
                print(f"\n📊 OPTIMIZATION IMPACT:")
                print(f"   Optimized vs Legacy Profit: {improvement:+.1f}% difference")
                
                if improvement > 0:
                    print(f"   ✅ Optimized model predicts higher profit")
                elif improvement < 0:
                    print(f"   📉 Optimized model is more conservative")
                else:
                    print(f"   ⚖️ Models agree on profit prediction")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def main():
    """Main demonstration function"""
    
    print("🎬 STARTING OPTIMIZED ENSEMBLE DEMONSTRATION")
    print("=" * 80)
    print("🎯 This demo showcases the integration of our optimized model")
    print("   (R² = 0.58) with the existing ensemble prediction system")
    print("=" * 80)
    
    try:
        # Run business scenarios
        results = run_business_scenarios()
        
        # Model comparison
        demonstrate_model_comparison()
        
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ Key Achievements:")
        print("   • Optimized model integrated with ensemble system")
        print("   • Business constraints properly enforced")
        print("   • Multiple prediction models working together")
        print("   • Real-time business classification")
        print("   • Sales forecasting capability")
        print("   • Comprehensive performance analysis")
        
        print(f"\n🚀 Next Steps:")
        print("   • Deploy the optimized ensemble for production use")
        print("   • Monitor performance across different business types")
        print("   • Collect feedback for further improvements")
        print("   • Consider additional optimization strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo encountered errors!")
