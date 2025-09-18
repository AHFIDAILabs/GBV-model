from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import base64
import io
import matplotlib
import warnings
import logging

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_utils import GBVVulnerabilityPredictor
from Explanation_engine import explain_instance

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = "gbv-prediction-app-2024"

# ------------------- Initialize predictor globally -------------------

try:
    predictor = GBVVulnerabilityPredictor()
    if not predictor.load_model():
        logging.error("Failed to load model at startup")
        predictor = None
    else:
        logging.info("Model loaded successfully at startup")
except Exception as e:
    logging.exception("Exception during predictor initialization: %s", e)
    predictor = None

# ------------------- Utility functions -------------------

def safe_int_cast(v, default=0):
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def synthesize_insights(top_features, top_n=5):
    reasons = []
    suggestions = []
    if not top_features:
        return {"reasons": reasons, "suggestions": suggestions}
    items = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    for feat, impact in items:
        direction = "increases" if impact > 0 else "decreases"
        reasons.append(f"{feat} {direction} vulnerability (impact: {impact:.3f})")
        if impact > 0:
            suggestions.append(f"Monitor {feat} closely and consider targeted support/intervention for factors related to {feat}.")
        else:
            suggestions.append(f"{feat} appears protective; promote/maintain factors associated with {feat} where appropriate.")
    return {"reasons": reasons, "suggestions": suggestions}

def create_shap_plot(shap_values, feature_names, feature_values):
    try:
        plt.figure(figsize=(8, max(2, len(shap_values) * 0.6)))
        shap_abs = np.abs(shap_values)
        order = np.argsort(shap_abs)[::-1]
        ordered_vals = shap_values[order]
        ordered_names = [feature_names[i] for i in order]
        colors = ["#ef4444" if v < 0 else "#2563eb" for v in ordered_vals]
        y_pos = list(range(len(ordered_vals)))
        plt.barh(y_pos, ordered_vals, color=colors, alpha=0.85)
        plt.yticks(y_pos, ordered_names)
        plt.xlabel("SHAP value (impact on prediction)")
        plt.axvline(0, color="#333", alpha=0.2)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return data
    except Exception as e:
        logging.exception("Failed to create SHAP plot: %s", e)
        try:
            plt.close()
        except Exception:
            pass
        return None

# ------------------- Routes -------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        message = request.form.get("message", "")
        logging.info(f"Received contact form: {name}, {email}, {message}")
        return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    global predictor
    if predictor is None:
        return jsonify({"success": False, "error": "Model not loaded."})
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid input format; expected JSON object."})
        model_input = {k: safe_int_cast(v, default=0) for k, v in data.items()}
        try:
            required = predictor.get_required_features()
            if isinstance(required, (list, tuple)):
                for feat in required:
                    if feat not in model_input:
                        model_input[feat] = 0
        except Exception:
            pass
        raw_pred = predictor.predict_single(model_input)
        if isinstance(raw_pred, dict):
            prediction = raw_pred
        else:
            pred_int = int(raw_pred) if raw_pred is not None else 0
            prediction = {
                "prediction": pred_int,
                "prediction_label": "VULNERABLE" if pred_int == 1 else "NOT VULNERABLE",
                "confidence": 100.0 if pred_int in (0, 1) else 0.0,
                "risk_level": "High" if pred_int == 1 else "Low"
            }
        try:
            explanation = explain_instance(predictor, model_input, top_n=10) or {}
        except Exception as e:
            logging.exception("explain_instance failed: %s", e)
            explanation = {}
        top_features = explanation.get("top_features") or {}
        insights = explanation.get("insights") or {"reasons": [], "suggestions": []}
        if not insights.get("reasons") and not insights.get("suggestions"):
            synthesized = synthesize_insights(top_features, top_n=10)
            insights["reasons"] = synthesized["reasons"]
            insights["suggestions"] = synthesized["suggestions"]
        explanation["insights"] = insights
        explanation["top_features"] = top_features
        shap_plot_base64 = None
        if top_features:
            try:
                feature_names = list(top_features.keys())
                shap_values = np.array(list(top_features.values())).astype(float)
                shap_plot_base64 = create_shap_plot(shap_values, feature_names, [model_input.get(fn, 0) for fn in feature_names])
            except Exception:
                shap_plot_base64 = None
        response = {
            "success": True,
            "prediction": prediction,
            "explanation": explanation,
            "shap_plot": shap_plot_base64,
            "user_input": model_input,
            "model_input": model_input,
        }
        logging.info("Prediction request handled: prediction=%s", prediction.get("prediction"))
        return jsonify(response)
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    if predictor is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "Service healthy"})

# ------------------- Main -------------------

if __name__ == "__main__":
    import threading, webbrowser, time
    def _open_browser():
        time.sleep(0.6)
        try:
            webbrowser.open("http://127.0.0.1:5000")
        except Exception:
            pass
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
