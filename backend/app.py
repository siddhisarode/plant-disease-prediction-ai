from fastapi import FastAPI
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from fastapi import UploadFile, File
import numpy as np
from backend.utils.image_utils import preprocess_image
from backend.utils.remedies import REMEDIES

from fastapi.middleware.cors import CORSMiddleware

# Attempt to import TensorFlow, but allow graceful degradation if it fails
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: TensorFlow import failed: {e}")
    print("API endpoints that require the model will return an error")
    TENSORFLOW_AVAILABLE = False
    tf = None

CLASS_NAMES = [
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria_Leaf_Spot"
]

CONFIDENCE_THRESHOLD = 0.65


app = FastAPI(title="Plant Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "..", "model", "plant_model.tflite")

# Use TFLite interpreter for lower memory / faster inference
interpreter = None
input_details = None
output_details = None

if TENSORFLOW_AVAILABLE:
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Warning: Failed to initialize TFLite interpreter: {e}")
        TENSORFLOW_AVAILABLE = False

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Plant Disease API is live",
        "model_loaded": TENSORFLOW_AVAILABLE and interpreter is not None
    }
@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        processed_image = preprocess_image(image_bytes)
    except Exception:
        return {"error": "invalid_image", "message": "Please upload a valid plant image."}

    return {
        "status": "success",
        "shape": processed_image.shape
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not TENSORFLOW_AVAILABLE or interpreter is None:
        return {
            "error": "Model not loaded",
            "message": "TensorFlow or the model failed to load. Please check server logs."
        }
    
    # Limit file size (5MB) for safety
    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > 5 * 1024 * 1024:
        return {"error": "File too large (max 5MB)"}

    image_bytes = await file.read()

    # Preprocess image
    try:
        processed_image = preprocess_image(image_bytes)
    except Exception:
        return {"error": "invalid_image", "message": "Please upload a valid plant image."}

    # Model prediction (TFLite)
    start = time.time()
    try:
        # Ensure dtype matches input
        dtype = input_details[0]["dtype"]
        interpreter.set_tensor(input_details[0]["index"], processed_image.astype(dtype))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
    except Exception as e:
        return {"error": "inference_failed", "message": str(e)}
    end = time.time()

    confidence = float(np.max(predictions))
    predicted_class = CLASS_NAMES[int(np.argmax(predictions))]

    # Low confidence handling
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Uncertain",
            "confidence": round(confidence * 100, 2),
            "message": "Model is not confident. Please upload a clearer image.",
            "inference_time": round(end - start, 3)
        }

    info = REMEDIES[predicted_class]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "description": info["description"],
        "remedies": info["remedies"],
        "care_tips": info["care_tips"],
        "inference_time": round(end - start, 3)
    }



