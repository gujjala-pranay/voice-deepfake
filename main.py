from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import torch
import librosa
import numpy as np
import os
import shutil
from src.model import DeepfakeDetectorCNN, get_model
from src.ensemble import create_ensemble
from utils import AudioDataset
import torch.nn.functional as F

app = FastAPI(title="Audio Deepfake Detection API")

# Load Model (default to enhanced)
model = None
current_model_type = None

def load_model_by_type(model_type='enhanced', ensemble_type='standard'):
    """Load model based on type"""
    if model_type == 'ensemble':
        model = create_ensemble(ensemble_type)
        model_path = f'model/ensemble_{ensemble_type}_model.pth'
    else:
        model = get_model(model_type)
        model_path = f'model/{model_type}_model.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize with enhanced model
model = load_model_by_type('enhanced')

def preprocess_audio(audio_path, max_len=64000):
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
    else:
        audio = audio[:max_len]
    
    # Use AudioDataset to extract all features
    dataset = AudioDataset.__new__(AudioDataset)  # Create instance without calling __init__
    features = dataset.extract_all_features(audio, sr)
    
    # Convert to tensors and add batch dimension
    spectral = torch.FloatTensor(features['spectral']).unsqueeze(0)
    mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0)
    phase = torch.FloatTensor(features['phase']).unsqueeze(0)
    
    return spectral, mfcc, phase

@app.get("/")
async def root():
    return {
        "message": "Audio Deepfake Detection API is running",
        "endpoints": {
            "predict": "/predict (POST)",
            "models": "/models (GET)",
            "docs": "/docs"
        }
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models and their configurations"""
    return {
        "available_models": [
            {
                "name": "enhanced",
                "description": "Full CNN with all features (Spectral, MFCC, Phase)",
                "speed": "Medium",
                "accuracy": "High",
                "features_used": 3
            },
            {
                "name": "lightweight",
                "description": "Fast model using only spectral features",
                "speed": "Fast",
                "accuracy": "Medium",
                "features_used": 1
            },
            {
                "name": "ensemble",
                "description": "Ensemble of multiple models",
                "speed": "Slow",
                "accuracy": "Very High",
                "features_used": 3,
                "ensemble_types": ["standard", "multiscale", "adaptive"]
            }
        ]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("enhanced", description="Model type: enhanced, lightweight, or ensemble"),
    ensemble_type: str = Query("standard", description="Ensemble type: standard, multiscale, or adaptive")
):
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav and .mp3 are supported.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        spectral, mfcc, phase = preprocess_audio(temp_path)
        
        # Load the selected model
        global model
        if model_type not in ['enhanced', 'lightweight', 'ensemble']:
            model_type = 'enhanced'
            
        selected_model = load_model_by_type(model_type, ensemble_type)
        
        with torch.no_grad():
            # Use appropriate features based on model type
            if model_type == 'lightweight':
                output = selected_model(spectral)
            else:
                output = selected_model(spectral, mfcc, phase)
            
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        result = {
            "filename": file.filename,
            "prediction": "REAL" if prediction.item() == 1 else "FAKE",
            "confidence": float(confidence.item()),
            "model_type": model_type,
            "ensemble_type": ensemble_type if model_type == 'ensemble' else None,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
