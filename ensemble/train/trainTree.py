import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

X = pd.read_csv("ensemble/data/X_scaled.csv")
y = pd.read_csv("ensemble/data/y_scaled.csv")

tree = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

tree.fit(X, y)
rf.fit(X, y)

os.makedirs("ensemble/models", exist_ok=True)
joblib.dump(tree, "ensemble/models/tree_model.pkl")
joblib.dump(rf, "ensemble/models/rf_model.pkl")
