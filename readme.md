- install requirements
```bash
pip install -r requirements.txt
```

- jalankan ml flow
```bash
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000
```

- training dan buat model baru
```bash
python ensemble/train/preprocess.py
python ensemble/train/trainMLP.py
python ensemble/train/trainTree.py
python ensemble/train/trainForecast.py
python ensemble/train/trainKmeans.py
python ensemble/train/trainFuzzy.py
```

- jalankan model baru
```bash
python ensemble/mapping/questionMap.py
python ensemble/predict/predictAll.py

atau menjalankan keduanya:
python ensemble/main.py
```