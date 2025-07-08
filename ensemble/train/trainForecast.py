import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os

df = pd.read_csv("../../dataset/csv/train.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Sort & filter
df = df.sort_values("Date")
df = df[df["Open"] == 1]

# Exogenous = Promo dan SchoolHoliday
exog = df[["Promo", "SchoolHoliday"]]
endog = df["Sales"]

model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
fit = model.fit(disp=False)

os.makedirs("../models", exist_ok=True)
joblib.dump(fit, "../models/arimax_model.pkl")
