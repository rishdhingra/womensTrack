#!/usr/bin/env python3
"""
EndoDetect AI - Radiomics-Only Backend API
Simplified API focused on radiomics-based inference
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import nibabel as nib
from pathlib import Path
from datetime import datetime
from scipy import ndimage, stats
from scipy.ndimage import label, center_of_mass

# Import radiomics classifier
sys.path.append(str(Path(__file__).parent))
from train_radiomics_simple import extract_basic_radiomics_features, RadiomicsClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Configuration
UPLOAD_FOLDER = Path('./uploads')
OUTPUT_FOLDER = Path('./outputs')
RADIOMICS_MODEL_PATH = Path('./models_radiomics/best_model.pth')
RADIOMICS_METADATA_PATH = Path('./models_radiomics/metadata.json')

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'dcm', 'nii', 'nii.gz', 'zip', 'png', 'jpg', 'jpeg'}

# Load radiomics model
device = torch.device('cpu')
radiomics_model = None
feature_names = None
model_metadata = {}

if RADIOMICS_MODEL_PATH.exists() and RADIOMICS_METADATA_PATH.exists():
    try:
        # Load metadata to get feature dimension
        with open(RADIOMICS_METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)
        
        feature_dim = model_metadata.get('radiomics_features', 30)
        feature_names = model_metadata.get('feature_names', [])
        
        # Initialize and load model
        radiomics_model = RadiomicsClassifier(input_dim=feature_dim).to(device)
        checkpoint = torch.load(RADIOMICS_MODEL_PATH, map_location=device)
        radiomics_model.load_state_dict(checkpoint['model_state_dict'])
        radiomics_model.eval()
        
        print(f"✅ Radiomics model loaded from {RADIOMICS_MODEL_PATH}")
        print(f"   Accuracy: {checkpoint.get('accuracy', 0)*100:.2f}%")
        print(f"   Features: {feature_dim}")
    except Exception as e:
        print(f"⚠️  Error loading radiomics model: {e}")
        print(f"   API will run but inference will fail")
else:
    print(f"⚠️  Radiomics model not found")
    print(f"   Train model first: python3 train_radiomics_simple.py")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": radiomics_model is not None,
        "model_type": "radiomics_classifier",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload imaging file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_id = f"{timestamp}_{filename}"
        filepath = UPLOAD_FOLDER / file_id
        file.save(str(filepath))
        
        return jsonify({
            "file_id": file_id,
            "filename": filename,
            "status": "uploaded",
            "timestamp": datetime.now().isoformat()
        }), 200
    
    return jsonify({"error": "Invalid file type"}), 400


@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Run radiomics-based inference"""
    if radiomics_model is None:
        return jsonify({"error": "Model not loaded. Please train the radiomics model first."}), 500
    
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({"error": "file_id required"}), 400
        
        filepath = UPLOAD_FOLDER / file_id
        if not filepath.exists():
            return jsonify({"error": "File not found"}), 404
        
        # Extract radiomics features
        features, feat_names = extract_basic_radiomics_features(filepath, None)
        
        if features is None:
            return jsonify({"error": "Failed to extract radiomics features"}), 500
        
        # Ensure feature dimension matches
        if len(features) != len(feature_names):
            # Pad or truncate if needed
            if len(features) < len(feature_names):
                features = np.pad(features, (0, len(feature_names) - len(features)), 'constant')
            else:
                features = features[:len(feature_names)]
        
        # Run inference
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
            prediction = radiomics_model(features_tensor)
            probability = prediction.item()
            has_endometriosis = probability > 0.5
        
        # Create results
        result_id = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {
            "result_id": result_id,
            "file_id": file_id,
            "prediction": {
                "has_endometriosis": bool(has_endometriosis),
                "probability": float(probability),
                "confidence": float(abs(probability - 0.5) * 2)  # 0-1 scale
            },
            "radiomics_features": {
                "count": len(features),
                "feature_names": feature_names,
                "feature_values": {name: float(val) for name, val in zip(feature_names, features)}
            },
            "model_info": {
                "type": "radiomics_classifier",
                "accuracy": model_metadata.get('best_accuracy_percentage', 0),
                "dataset": model_metadata.get('dataset', 'N/A')
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        result_path = OUTPUT_FOLDER / f"{result_id}.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Retrieve saved results"""
    result_path = OUTPUT_FOLDER / f"{result_id}.json"
    
    if not result_path.exists():
        return jsonify({"error": "Results not found"}), 404
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results), 200


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if radiomics_model is None:
        return jsonify({
            "status": "Model not loaded",
            "message": "Please train the radiomics model first"
        }), 404
    
    return jsonify(model_metadata), 200


if __name__ == '__main__':
    print("=" * 60)
    print("EndoDetect AI - Radiomics-Only Backend API")
    print("=" * 60)
    print(f"Model: {'✅ Loaded' if radiomics_model else '❌ Not loaded'}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 60)
    print("\nStarting server on http://localhost:5001")
    print("API Endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/upload - Upload imaging file")
    print("  POST /api/inference - Run radiomics inference")
    print("  GET  /api/results/<id> - Retrieve results")
    print("  GET  /api/model-info - Get model metadata")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
