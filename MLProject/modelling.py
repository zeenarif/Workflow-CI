import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import os

# ── Path dataset ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'heart_disease_uci_preprocessing')
TRAIN_PATH = os.path.join(DATA_DIR, 'heart_disease_train.csv')
TEST_PATH  = os.path.join(DATA_DIR, 'heart_disease_test.csv')

# ── Load data ─────────────────────────────────────────────
print("Loading dataset...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test  = test_df.drop(columns=['target'])
y_test  = test_df['target']

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ── MLflow setup (lokal, tanpa server) ────────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("heart_disease_classification")

# ── Training dengan autolog ───────────────────────────────
print("\nMemulai training...")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_CI"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')

    print("\nHasil Evaluasi:")
    print(f"   Accuracy  : {accuracy:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Sehat', 'Sakit']))

print("\nTraining selesai!")
