"""
modelling.py
Versi CI - dijalankan oleh GitHub Actions via MLflow Project.
"""

import matplotlib
matplotlib.use('Agg')

import mlflow
import mlflow.sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Setup MLflow ──────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("iris-ci")

# ── Load Data ─────────────────────────────────────
X_train = np.load('iris_preprocessing/X_train.npy')
X_test  = np.load('iris_preprocessing/X_test.npy')
y_train = np.load('iris_preprocessing/y_train.npy')
y_test  = np.load('iris_preprocessing/y_test.npy')

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ── Training ──────────────────────────────────────
with mlflow.start_run(run_name="RF-CI"):
    params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec  = recall_score(y_test, preds, average='macro')
    f1   = f1_score(y_test, preds, average='macro')

    mlflow.log_params(params)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.sklearn.log_model(model, "model")

    # Simpan model sebagai file artifact
    os.makedirs("outputs", exist_ok=True)
    import pickle
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("outputs/model.pkl")

    print(f"acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

print("Training selesai!")
