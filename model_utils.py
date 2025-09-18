import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.exceptions import NotFittedError
import warnings

warnings.filterwarnings("ignore")


class GBVVulnerabilityPredictor:
    """
    A class to handle GBV vulnerability predictions using the trained logistic regression model
    """

    def __init__(self, model_dir=None):
        """
        Initialize the predictor with model directory
        """
        if model_dir is None:
            # Relative path to the "model" folder in the repo
            model_dir = os.path.join(os.path.dirname(__file__), "model")
        self.model_dir = model_dir
        self.model = None
        self.top_features = None
        self.feature_importance = None
        self._is_loaded = False

    def load_model(self):
        """
        Load the trained model and associated artifacts
        """
        try:
            # Load the trained model
            model_path = os.path.join(self.model_dir, "gbv_logistic_regression_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")

            # Load top features
            features_path = os.path.join(self.model_dir, "top_features.joblib")
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Features file not found: {features_path}")
            self.top_features = joblib.load(features_path)
            print(f"Top features loaded: {len(self.top_features)} features")

            # Load feature importance (optional)
            importance_path = os.path.join(self.model_dir, "feature_importance.csv")
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)
                print("Feature importance loaded successfully")

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict_single(self, input_data):
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")

        try:
            input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
            missing_features = set(self.top_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            input_df = input_df[self.top_features]

            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            confidence = round(probabilities[prediction] * 100, 2)

            return {
                'prediction': int(prediction),
                'prediction_label': 'Vulnerable' if prediction == 1 else 'Not Vulnerable',
                'confidence': confidence,
                'probabilities': {
                    'not_vulnerable': round(probabilities[0] * 100, 2),
                    'vulnerable': round(probabilities[1] * 100, 2)
                },
                'risk_level': self._get_risk_level(probabilities[1])
            }

        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def predict_batch(self, input_data):
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")

        try:
            input_df = pd.DataFrame(input_data) if isinstance(input_data, list) else input_data.copy()
            missing_features = set(self.top_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            input_df = input_df[self.top_features]

            predictions = self.model.predict(input_df)
            probabilities = self.model.predict_proba(input_df)

            results = []
            for i in range(len(predictions)):
                pred = predictions[i]
                probs = probabilities[i]
                confidence = round(probs[pred] * 100, 2)
                results.append({
                    'prediction': int(pred),
                    'prediction_label': 'Vulnerable' if pred == 1 else 'Not Vulnerable',
                    'confidence': confidence,
                    'probabilities': {
                        'not_vulnerable': round(probs[0] * 100, 2),
                        'vulnerable': round(probs[1] * 100, 2)
                    },
                    'risk_level': self._get_risk_level(probs[1])
                })

            return results

        except Exception as e:
            raise ValueError(f"Batch prediction error: {str(e)}")

    def _get_risk_level(self, vulnerability_probability):
        if vulnerability_probability < 0.3:
            return "Low"
        elif vulnerability_probability < 0.5:
            return "Medium"
        elif vulnerability_probability < 0.75:
            return "High"
        else:
            return "Very High"

    def get_feature_importance(self, top_n=None):
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        return self.feature_importance.head(top_n) if top_n else self.feature_importance

    def get_required_features(self):
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")
        return self.top_features.copy()

    def validate_input(self, input_data):
        validation_result = {'is_valid': True, 'messages': [], 'missing_features': [], 'extra_features': []}
        if not self._is_loaded:
            validation_result['is_valid'] = False
            validation_result['messages'].append("Model not loaded")
            return validation_result

        input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
        missing_features = set(self.top_features) - set(input_df.columns)
        extra_features = set(input_df.columns) - set(self.top_features)

        if missing_features:
            validation_result['is_valid'] = False
            validation_result['missing_features'] = list(missing_features)
            validation_result['messages'].append(f"Missing required features: {missing_features}")
        if extra_features:
            validation_result['extra_features'] = list(extra_features)
            validation_result['messages'].append(f"Extra features (will be ignored): {extra_features}")

        return validation_result


# ----------------- Utility Functions -----------------

def load_model(model_dir=None):
    predictor = GBVVulnerabilityPredictor(model_dir)
    if predictor.load_model():
        return predictor
    return None

def create_prediction_template(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
    features_path = os.path.join(model_dir, "top_features.joblib")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    top_features = joblib.load(features_path)
    return {feature: 0 for feature in top_features}

def get_model_info(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
    info = {'model_available': False, 'features_count': 0, 'required_features': [], 'feature_importance_available': False}
    model_path = os.path.join(model_dir, "gbv_logistic_regression_model.joblib")
    info['model_available'] = os.path.exists(model_path)
    features_path = os.path.join(model_dir, "top_features.joblib")
    if os.path.exists(features_path):
        features = joblib.load(features_path)
        info['features_count'] = len(features)
        info['required_features'] = features
    importance_path = os.path.join(model_dir, "feature_importance.csv")
    info['feature_importance_available'] = os.path.exists(importance_path)
    return info

def example_prediction():
    predictor = load_model()
    if predictor is None:
        print("Failed to load predictor")
        return None
    sample_input = create_prediction_template()
    sample_input.update({
        'individual_age': 25,
        'gender': 0,
        'individual_employment_status': 1,
        'current_living_arrangement': 1,
        'educational_status': 3,
    })
    result = predictor.predict_single(sample_input)
    print("Prediction Result:")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probabilities: {result['probabilities']}")
    return result


if __name__ == "__main__":
    print("Testing GBV Model Utilities")
    print("="*40)
    model_info = get_model_info()
    print("Model Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    if model_info['model_available']:
        print("\nRunning example prediction...")
        example_prediction()
    else:
        print("\nModel not available. Please run train.py first.")
