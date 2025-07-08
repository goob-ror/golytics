import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

df = pd.read_csv("../../dataset/csv/processed_model_data.csv")

# Fitur clustering
features = df[["Has Website", "Social Media Presence", "Marketplace Usage",
               "Payment Digital Adoption", "E-Wallet Acceptance"]].astype(float)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

os.makedirs("../models", exist_ok=True)
joblib.dump(kmeans, "../models/kmeans_model.pkl")
