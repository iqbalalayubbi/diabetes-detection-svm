from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from database import save_prediction, get_all_predictions
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import locale

# Load .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# 1. Load model dan artefak preprocessing
saved = joblib.load("svm_model_v2.pkl")
svm = saved["model"]
scalers = saved.get("scaler")
encoders = saved.get("encoders")
obj_cols_tr = saved.get("obj_cols")
num_cols_tr = saved.get("num_cols")
feature_order = saved.get("feature_order")

print(f"Model loaded - scalers type: {type(scalers)}, encoders type: {type(encoders)}")
print(f"obj_cols_tr: {obj_cols_tr}")
print(f"num_cols_tr: {num_cols_tr}")
print(f"feature_order: {feature_order}")

if feature_order is not None:
    feature_order = list(feature_order)
    feature_order = [f for f in feature_order if f != "class"]


# Helper encoding
def safe_add_unseen_and_transform(le: LabelEncoder, s: pd.Series):
    vals = s.astype(str)
    unseen = sorted(set(vals.unique()) - set(le.classes_))
    if unseen:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=object)])
    return le.transform(vals)

# Endpoint Prediksi
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         if not isinstance(data, dict):
#             return jsonify({"error": "Request harus berupa JSON object"}), 400

#         print(f"Received data: {data}")  # Debug log
#         print(f"Data types: {[(k, type(v).__name__) for k, v in data.items()]}")  # Debug data types
#         df = pd.DataFrame([data])
#         print(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")  # Debug log
#         print(f"DataFrame dtypes: {df.dtypes.to_dict()}")  # Debug DataFrame types

#         # Tentukan kolom kategorikal & numerik
#         obj_cols = list(obj_cols_tr) if obj_cols_tr is not None else (list(encoders.keys()) if encoders else [])
#         num_cols = list(num_cols_tr) if num_cols_tr is not None else df.select_dtypes(include=[np.number]).columns.tolist()

#         # Skip encoding and scaling since they're None in the model file
#         # The model was likely trained on already preprocessed data
#         print(f"Data before prediction: {df.values}")
#         print(f"Data shape: {df.shape}")
#         print(f"Column order: {list(df.columns)}")

#         # Urutkan fitur sesuai training
#         if feature_order and len(feature_order) > 0:
#             final_cols = [c for c in feature_order if c in df.columns]
#             if final_cols:  # Only reorder if we have valid columns
#                 df = df[final_cols]

#         # Validasi bahwa kita memiliki kolom yang diperlukan untuk prediksi
#         if df.empty or len(df.columns) == 0:
#             return jsonify({"error": "No valid features for prediction after preprocessing", "success": False}), 400

#         # Prediksi
#         try:
#             pred = svm.predict(df)
#             result = int(pred[0])

#             # Calculate dynamic confidence using decision function
#             try:
#                 if hasattr(svm, 'decision_function'):
#                     # Use decision function to calculate confidence based on distance from boundary
#                     decision_score = svm.decision_function(df)
#                     # Convert decision score to confidence (0-1 range)
#                     # Higher absolute decision score = higher confidence
#                     confidence = 1 / (1 + np.exp(-abs(decision_score[0])))
#                     confidence = float(min(confidence, 0.999))  # Cap at 99.9%
#                 elif hasattr(svm, 'predict_proba'):
#                     proba = svm.predict_proba(df)
#                     confidence = float(proba[0][result])
#                 else:
#                     # Fallback: use a moderate confidence
#                     confidence = 0.85
#             except Exception as e:
#                 print(f"Warning: Could not calculate confidence: {e}")
#                 confidence = 0.85  # Fallback confidence

#             print(f"Calculated confidence: {confidence}")

#         except Exception as e:
#             return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 500

#         # Generate unique ID and timestamp
#         prediction_id = str(uuid.uuid4())[:16]  # Short unique ID

#         # Set locale for Indonesian timestamp
#         try:
#             locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
#         except:
#             locale.setlocale(locale.LC_TIME, '')  # Use default locale

#         timestamp = datetime.now().strftime("%d %B %Y %H:%M WIB")

#         # Convert prediction to readable text
#         prediction_text = "Diabetes" if result == 1 else "Tidak Diabetes"

#         # Generate suggestion based on prediction
#         if result == 1:
#             suggestion = "Berdasarkan gejala yang Anda alami, terdapat indikasi diabetes. Sebaiknya konsultasikan dengan dokter untuk pemeriksaan lebih lanjut dan tes gula darah."
#         else:
#             suggestion = "Berdasarkan gejala yang Anda alami, tidak terdapat indikasi diabetes yang signifikan. Namun tetap jaga pola hidup sehat."

#         print(f"Prediction: {result}, Text: {prediction_text}")
#         print(f"All input values: {data}")

#         # Simpan ke database dengan informasi tambahan
#         # save_prediction(data, pred, prediction_id, prediction_text, confidence)

#         return jsonify({
#             "success": True,
#             "id": prediction_id,
#             "prediction": prediction_text,
#             "confidence": confidence,
#             "suggestion": suggestion,
#             "timestamp": timestamp,
#             "numeric_prediction": result  # Keep for backward compatibility
#         })

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Request harus berupa JSON object"}), 400

    data = pd.DataFrame([data])

    for col, le in encoders.items():
        if col in data.columns:
            data[col] = safe_add_unseen_and_transform(le, data[col])

    for col, sc in scalers.items():
        if col in data.columns:
            data[col] = sc.transform(data[[col]])
    
    data = data[feature_order]



    pred = svm.predict(data)[0]

    prediction_id = str(uuid.uuid4())[:16]  # Short unique ID
    timestamp = datetime.now().strftime("%d %B %Y %H:%M WIB")


    return jsonify({
        "success": True,
        "id": prediction_id,
        "prediction": pred,
        # "confidence": confidence,
        # "suggestion": suggestion,
        "timestamp": timestamp,
        # "numeric_prediction": result  # Keep for backward compatibility
    })


# Endpoint Riwayat Prediksi
@app.route("/history", methods=["GET"])
def history():
    records = get_all_predictions()
    for r in records:
        r["_id"] = str(r["_id"])
    return jsonify(records)

@app.route("/predict-ui", methods=["GET"])
def predict_ui():
    fields = feature_order if feature_order is not None else []
    return render_template("predict_ui.html", fields=fields)

# Run Server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Running Flask server on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)