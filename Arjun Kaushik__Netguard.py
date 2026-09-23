import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier
import joblib

# ---------------------------------------------------------------------------
# 1. Column names (NSL-KDD files have no header row)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

TRAIN_PATH = r"C:\Users\HP\Desktop\IOT attack detection\KDDTrain+.txt"
TEST_PATH = r"C:\Users\HP\Desktop\IOT attack detection\KDDTest+.txt"

def load_data(path):
    df = pd.read_csv(path, names=COLUMN_NAMES)
    df = df.drop(columns=["difficulty"])
    return df


def binarize_label(df):
    # NSL-KDD labels are specific attack names (e.g. "neptune", "smurf", "normal").
    # Collapse to binary: 0 = normal, 1 = attack.
    df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)
    return df


def preprocess(train_df, test_df):
    # One-hot encode categoricals. Fit encoding on combined columns so
    # train/test end up with identical dummy columns.
    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS)

    train_enc = combined.xs("train")
    test_enc = combined.xs("test")

    X_train = train_enc.drop(columns=["label"])
    y_train = train_enc["label"]
    X_test = test_enc.drop(columns=["label"])
    y_test = test_enc["label"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


def evaluate(name, y_true, y_pred):
    print(f"\n{'=' * 50}")
    print(f"{name} Results")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["normal", "attack"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    return {"model": name, "precision": precision, "recall": recall, "f1": f1}


def print_feature_importance(name, model, feature_names, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} features — {name}")
    for i in idx:
        print(f"  {feature_names[i]:<35} {importances[i]:.4f}")


def main():
    print("Loading data...")
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    train_df = binarize_label(train_df)
    test_df = binarize_label(test_df)

    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")

    X_train, y_train, X_test, y_test, scaler = preprocess(train_df, test_df)
    feature_names = list(X_train.columns)

    # ---- Random Forest ----
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_scores = evaluate("Random Forest", y_test, rf_preds)
    print_feature_importance("Random Forest", rf, feature_names)

    # ---- XGBoost ----
    print("\nTraining XGBoost...")
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_scores = evaluate("XGBoost", y_test, xgb_preds)
    print_feature_importance("XGBoost", xgb, feature_names)

    # ---- Side-by-side comparison ----
    print(f"\n{'=' * 50}")
    print("Model Comparison (macro-averaged)")
    print("=" * 50)
    comp_df = pd.DataFrame([rf_scores, xgb_scores]).set_index("model")
    print(comp_df.round(4))

    # ---- Save models ----
    joblib.dump(rf, "random_forest_ids.joblib")
    joblib.dump(xgb, "xgboost_ids.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_names, "feature_names.joblib")
    print("\nSaved: random_forest_ids.joblib, xgboost_ids.joblib, scaler.joblib, feature_names.joblib")


if __name__ == "__main__":
    main()