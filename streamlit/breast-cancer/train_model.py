import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ==========================================================
# Configuration
# ==========================================================

DATASET = "breast_cancer.csv"
MODEL_FILE = "breast_cancer_pipeline.pkl"
DEFAULTS_FILE = "defaults.json"

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(DATASET)

print(f"Dataset shape: {df.shape}")

# ==========================================================
# Clean Dataset
# ==========================================================

# Remove unused columns if they exist
for column in ["id", "Unnamed: 32"]:
    if column in df.columns:
        df = df.drop(columns=column)

# Encode diagnosis
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

print(f"Features: {len(df.columns)-1}")

# ==========================================================
# Save Example Patient
# ==========================================================

example_patient = df.drop(columns=["diagnosis"]).iloc[0].to_dict()

with open(DEFAULTS_FILE, "w") as f:
    json.dump(example_patient, f, indent=4)

print(f"Created {DEFAULTS_FILE}")

# ==========================================================
# Split Dataset
# ==========================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ==========================================================
# Build Pipeline
# ==========================================================

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("classifier", LogisticRegression(C=0.1, max_iter=500, random_state=42)),
    ]
)

# ==========================================================
# Train
# ==========================================================

print("Training model...")

pipeline.fit(X_train, y_train)

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(pipeline, MODEL_FILE)

print(f"Saved model to '{MODEL_FILE}'")

# ==========================================================
# Quick Evaluation
# ==========================================================

accuracy = pipeline.score(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nDone!")
