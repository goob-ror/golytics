import pandas as pd
import json
import os

df = pd.read_csv("../../dataset/csv/processed_model_data.csv")

def fuzzy_rule(row):
    pemasukan = row["Monthly Revenue_>30 Juta"]
    pengeluaran = row["Monthly Revenue_<5 Juta"]
    if pemasukan == 1.0:
        return "sangat_menguntungkan"
    elif row["Monthly Revenue_5-15 Juta"] == 1.0:
        return "cukup_menguntungkan"
    elif pengeluaran == 1.0:
        return "rugi"
    return "tidak_diketahui"

results = []
for _, row in df.iterrows():
    label = fuzzy_rule(row)
    results.append({
        "id": _,
        "kategori": label
    })

os.makedirs("../models", exist_ok=True)
with open("../models/fuzzy_rules_output.json", "w") as f:
    json.dump(results, f, indent=2)
