from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import asyncio

app = FastAPI(title="Pneumonia Detection API", description="CNN Microservice for MERN Stack")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- CONFIGURATION ---
MODEL_PATH  = "model/pneumonia_densenet.keras"
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Load model on startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Pydantic model for JSON requests (if Express sends a URL)
class ScanRequest(BaseModel):
    imageURL: str

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesses raw image bytes for the DenseNet121 model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image) / 255.0  # Rescale exactly as in your ImageDataGenerator
    return np.expand_dims(arr, axis=0)

def get_prediction(img_array: np.ndarray) -> dict:
    """Handles the binary sigmoid prediction logic."""
    raw_score = model.predict(img_array, verbose=0)[0][0] # Extracts the single float
    
    # Sigmoid logic: >= 0.5 is Class 1 (Pneumonia), < 0.5 is Class 0 (Normal)
    is_pneumonia = float(raw_score) >= 0.5
    confidence = float(raw_score) if is_pneumonia else 1.0 - float(raw_score)
    
    return {
        "prediction": CLASS_NAMES[1] if is_pneumonia else CLASS_NAMES[0],
        "confidenceScore": round(confidence * 100, 2)
    }

@app.get("/health")
def health_check():
    """Endpoint for your Node.js server to check if Python is running."""
    return {"status": "ok", "model_loaded": model is not None}

# --- ENDPOINT 1: IF EXPRESS SENDS A URL (JSON) ---
@app.post("/predict-url")
async def predict_from_url(scan: ScanRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # Async HTTP request to fetch the image
        async with httpx.AsyncClient() as client:
            response = await client.get(scan.imageURL, timeout=10)
            response.raise_for_status()
            
            # Content-type validation
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="URL does not point to an image.")
                
        # Preprocessing can stay synchronous as it is fast
        img_array = preprocess_image(response.content)
        
        # Offload the heavy model prediction to a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_prediction, img_array)
        
        return result
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: IF REACT/EXPRESS SENDS A FILE (FormData) ---
@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        
        # Offload the heavy model prediction to a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_prediction, img_array)
        
        result["filename"] = file.filename
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)