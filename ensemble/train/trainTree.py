import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

X = pd.read_csv("../data/X_scaled.csv")
y = pd.read_csv("../data/y_scaled.csv")

tree = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

tree.fit(X, y)
rf.fit(X, y)

os.makedirs("../models", exist_ok=True)
joblib.dump(tree, "../models/tree_model.pkl")
joblib.dump(rf, "../models/rf_model.pkl")
