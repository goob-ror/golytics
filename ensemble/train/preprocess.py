import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

DATA_PATH = "dataset/csv/processed_model_data.csv"
SAVE_PATH = "../data"

def preprocess_and_save():
    df = pd.read_csv(DATA_PATH)

    # Kolom input numerik untuk X
    input_cols = ["Has Website", "Social Media Presence", "Marketplace Usage",
                  "Payment Digital Adoption", "POS (Point of Sales) Usage",
                  "Online Ads Usage", "E-Wallet Acceptance"]

    # Ubah ke float (sudah one-hot sebagian)
    X = df[input_cols].astype(float)
    y = pd.DataFrame({
        "modal": df["Monthly Revenue_5-15 Juta"],     # dummy asumsi (karena tidak ada label angka)
        "profit": df["Monthly Revenue_>30 Juta"],
        "rugi": df["Monthly Revenue_<5 Juta"]
    })

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Normalisasi
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    pd.DataFrame(X_scaled, columns=X.columns).to_csv(f"{SAVE_PATH}/X_scaled.csv", index=False)
    pd.DataFrame(y_scaled, columns=y.columns).to_csv(f"{SAVE_PATH}/y_scaled.csv", index=False)

    joblib.dump(scaler_x, f"{SAVE_PATH}/scaler_x.pkl")
    joblib.dump(scaler_y, f"{SAVE_PATH}/scaler_y.pkl")

if __name__ == "__main__":
    preprocess_and_save()
