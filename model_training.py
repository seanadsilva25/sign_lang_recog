import pandas as pd
import pickle
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


#  CONFIGURATION

CSV_PATH   = "isl_dataset.csv"    
MODEL_PATH = "model/isl_model.pkl"    

# Random Forest hyperparameters (safe defaults for beginners)
N_ESTIMATORS      = 300    # number of trees
MAX_DEPTH         = 10    # limits tree depth
MIN_SAMPLES_SPLIT = 5     # minimum samples to split a node
NOISE_FACTOR      = 0.02  # small noise added to training data
RANDOM_STATE = 42      # for reproducible results
TEST_SIZE    = 0.2     # 20% of data used for evaluation

#  STEP 1 — Load the CSV dataset
def load_dataset(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[ERROR] Dataset not found. Run stage1_data_collection.py first.")
        return None, None

    print(f"[INFO] Loaded {len(df):,} rows across {df['label'].nunique()} words")

    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


#  STEP 2 — Normalize landmarks
def normalize_features(X_train, X_test):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, scaler


#  STEP 3 — Add noise to training data
def augment_with_noise(X_train, y_train):
    """
    Add small random noise to training data.
    Simulates natural hand variation and prevents memorization.
    Noisy copies are added on top of original — doubling training size.
    """
    noise       = np.random.normal(0, NOISE_FACTOR, X_train.shape)
    X_noisy     = X_train + noise
    X_augmented = np.vstack([X_train, X_noisy])
    y_augmented = np.concatenate([y_train, y_train])
    print(f"[INFO] Training samples after augmentation: {len(y_augmented)}")
    return X_augmented, y_augmented


def split_data(X, y, test_size: float = TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


#  STEP 3 — Train the Random Forest model
def train_model(X_train, y_train) -> RandomForestClassifier:
    print("[INFO] Training model...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return model


#  STEP 4 — Evaluate model accuracy
def evaluate_model(model: RandomForestClassifier, X_test, y_test):
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print("=" * 50)
    print(classification_report(y_test, y_pred, zero_division=0))


#  STEP 5 — Save the trained model
def save_model(model: RandomForestClassifier, model_path: str):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved → {model_path}")


#  Main pipeline
def main():
    print("\n" + "=" * 50)
    print("  ISL — STAGE 2: MODEL TRAINING")
    print("=" * 50 + "\n")

    # 1. Load data
    X, y = load_dataset(CSV_PATH)
    if X is None:
        return

    # 2. Train / test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Normalize features
    X_train, X_test, scaler = normalize_features(X_train, X_test)

    # 4. Augment training data with noise
    X_train, y_train = augment_with_noise(X_train, y_train)

    # 5. Train
    model = train_model(X_train, y_train)

    # 6. Evaluate
    evaluate_model(model, X_test, y_test)

    # 7. Save model AND scaler together (scaler needed at prediction time)
    save_model({"model": model, "scaler": scaler}, MODEL_PATH)

    print("\n[DONE] Stage 2 complete. Use isl_model.pkl for inference.")


if __name__ == "__main__":
    main()