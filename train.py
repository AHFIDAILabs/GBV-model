import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(data_path):
    """Load and prepare the GBV data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)

    target_col = "vulnerability_target"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in data")

    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    print(f" Target distribution:\n{df[target_col].value_counts()}")

    return df, target_col


def train_catboost_model(X_train, y_train, X_val, y_val, cat_features=None):
    """Train CatBoost model with best settings"""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="F1",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).astype(int)  # Ensure numeric
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)

    print(f"\n✅ {model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")

    return {"accuracy": accuracy, "f1": f1, "auc": auc, "logloss": ll}


def save_model_artifacts(model, feature_importance, model_dir="model"):
    """Save CatBoost model and related artifacts"""
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model_path = f"{model_dir}/gbv_catboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save feature importance
    feature_importance.to_csv(f"{model_dir}/feature_importance.csv", index=False)
    print(f"✅ Feature importance saved to {model_dir}/feature_importance.csv")

    print(f"✅ All artifacts saved to {model_dir}/ directory")


def create_prediction_template(features, model_dir="model"):
    """Create a prediction template file"""
    template_data = {feature: 0 for feature in features}
    template_df = pd.DataFrame([template_data])
    template_df.to_csv(f"{model_dir}/prediction_template.csv", index=False)

    print(f"✅ Prediction template saved to {model_dir}/prediction_template.csv")
    print("   Use this template to understand required input format for predictions")


def main():
    """Main training function"""

    print("🚀 Starting GBV Vulnerability Prediction - CatBoost")
    print("=" * 60)

    # Load environment variables
    load_dotenv() if os.path.exists('.env') else None

    # Config
    DATA_PATH = os.getenv("DATA_PATH", r"C:\Users\hp\Desktop\gbv_projects\data\cldataRbal_data.csv")
    MODEL_DIR = os.getenv("MODEL_DIR", "model")

    try:
        # 1. Load data
        df, target_col = load_and_prepare_data(DATA_PATH)

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Identify categorical columns (for CatBoost)
        cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

        # 2. Train/test split
        print("\n Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"✅ Training shape: {X_train.shape}")
        print(f"✅ Testing shape: {X_test.shape}")

        # Further split train into train/val
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # 3. Train CatBoost
        print("\n Training CatBoost model...")
        model = train_catboost_model(X_train_sub, y_train_sub, X_val, y_val, cat_features)

        # 4. Evaluate
        print("\n Evaluating model on test set...")
        metrics = evaluate_model(model, X_test, y_test, "CatBoost")

        # 5. Feature importance
        print("\n Extracting feature importance...")
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.get_feature_importance()
        }).sort_values(by="Importance", ascending=False)
        print(feature_importance.head(10))

        # 6. Save artifacts
        print("\n Saving model artifacts...")
        save_model_artifacts(model, feature_importance, MODEL_DIR)

        # 7. Create prediction template
        print("\n Creating prediction template...")
        create_prediction_template(X.columns, MODEL_DIR)

        print("\n Training completed successfully!")
        print("=" * 60)
        print(f" Model saved in: {MODEL_DIR}/")
        print(f" Final F1 Score: {metrics['f1']:.4f}")
        print(f" Final Accuracy: {metrics['accuracy']:.4f}")
        print(f" Final ROC AUC: {metrics['auc']:.4f}")

    except Exception as e:
        print(f" Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
