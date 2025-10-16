from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Ambil konfigurasi dari .env
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "diabetes_svm")  # default
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "predict_logs")  # default

# Koneksi ke MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Fungsi untuk menyimpan hasil prediksi
def save_prediction(input_data, prediction, prediction_id=None, prediction_text=None, confidence=None):
    doc = {
        "input": input_data,
        "prediction": int(prediction[0]) if hasattr(prediction, "__iter__") else int(prediction),
        "prediction_text": prediction_text or ("Diabetes" if prediction[0] == 1 else "Tidak Diabetes"),
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    if prediction_id:
        doc["id"] = prediction_id

    collection.insert_one(doc)
    return doc

# Fungsi untuk menampilkan log prediksi
def get_all_predictions(limit=10):
    return list(collection.find().sort("_id", -1).limit(limit))

print(f"Connected to MongoDB Atlas!")
