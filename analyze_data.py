#!/usr/bin/env python3
"""
Analyze the transaction data to understand the issues
"""

import json
import os
import numpy as np
import pandas as pd

def analyze_data():
    """Analyze the transaction data"""
    print("🔍 Analyzing Transaction Data")
    print("=" * 50)
    
    # Load data
    data_dir = 'generate/dataset/numeric/lanjutan'
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename.startswith('dataset_'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                all_data.extend(data)
    
    print(f"📊 Total records: {len(all_data)}")
    
    # Extract features
    modals = [d['modal_awal'] for d in all_data]
    profits = [d['profit'] for d in all_data]
    rugi = [d['rugi'] for d in all_data]
    pemasukan = [d['total_pemasukan'] for d in all_data]
    pengeluaran = [d['total_pengeluaran'] for d in all_data]
    
    print("\n📈 Data Ranges:")
    print(f"Modal: {min(modals):,} to {max(modals):,}")
    print(f"Profit: {min(profits):,} to {max(profits):,}")
    print(f"Rugi: {min(rugi):,} to {max(rugi):,}")
    print(f"Pemasukan: {min(pemasukan):,} to {max(pemasukan):,}")
    print(f"Pengeluaran: {min(pengeluaran):,} to {max(pengeluaran):,}")
    
    # Check for negative values
    negative_modals = [m for m in modals if m < 0]
    negative_profits = [p for p in profits if p < 0]
    negative_rugi = [r for r in rugi if r < 0]
    
    print(f"\n⚠️ Data Quality Issues:")
    print(f"Negative modals: {len(negative_modals)}")
    print(f"Negative profits: {len(negative_profits)}")
    print(f"Negative rugi: {len(negative_rugi)}")
    
    # Check profit/rugi logic
    inconsistent = 0
    for d in all_data:
        net = d['total_pemasukan'] - d['total_pengeluaran']
        if net > 0:  # Should have profit, no rugi
            if d['profit'] <= 0 or d['rugi'] > 0:
                inconsistent += 1
        else:  # Should have rugi, no profit
            if d['rugi'] <= 0 or d['profit'] > 0:
                inconsistent += 1
    
    print(f"Inconsistent profit/rugi logic: {inconsistent}")
    
    # Sample some records
    print(f"\n📋 Sample Records:")
    for i in range(min(5, len(all_data))):
        d = all_data[i]
        net = d['total_pemasukan'] - d['total_pengeluaran']
        print(f"Record {i+1}:")
        print(f"  Modal: {d['modal_awal']:,}")
        print(f"  Pemasukan: {d['total_pemasukan']:,}")
        print(f"  Pengeluaran: {d['total_pengeluaran']:,}")
        print(f"  Net: {net:,}")
        print(f"  Profit: {d['profit']:,}")
        print(f"  Rugi: {d['rugi']:,}")
        print()
    
    return all_data

if __name__ == "__main__":
    analyze_data()
