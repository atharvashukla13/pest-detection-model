#!/usr/bin/env python3
"""
Flask API for Filtered Pest Detection Model
Supports Rice, Wheat, Sugarcane, and Cotton pest detection
"""

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import json
import os
from PIL import Image
import io
import numpy as np
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow for memory optimization
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

class PestModelManager:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.input_shape = (224, 224, 3)
        self.load_model()
    
    def load_model(self):
        """Load the trained pest detection model with memory optimization"""
        try:
            logger.info("Loading pest detection model...")
            
            # Load model with optimizations
            self.model = tf.keras.models.load_model(
                'filtered_pest_model_best.h5', 
                compile=False,
                options=tf.saved_model.LoadOptions(experimental_io_device='/cpu:0')
            )
            
            # Build model with sample input
            sample_input = tf.random.normal((1,) + self.input_shape)
            _ = self.model(sample_input)
            
            # Compile model with minimal configuration
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load class names from model info
            with open('filtered_pest_model_info.json', 'r') as f:
                model_info = json.load(f)
                self.class_names = model_info['class_names']
            
            # Clear memory
            gc.collect()
            
            logger.info(f"✅ Pest detection model loaded successfully!")
            logger.info(f"   Classes: {len(self.class_names)}")
            logger.info(f"   Input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"❌ Error loading pest model: {e}")
            self.model = None
            self.class_names = None
    
    def preprocess_image(self, image_file):
        """Preprocess uploaded image for model prediction"""
        try:
            # Open and resize image
            img = Image.open(image_file).convert('RGB')
            img = img.resize(self.input_shape[:2])
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_pest(self, image_file):
        """Predict pest type from image with memory optimization"""
        if not self.model:
            return None, "Model not loaded"
        
        # Preprocess image
        processed_image = self.preprocess_image(image_file)
        if processed_image is None:
            return None, "Error preprocessing image"
        
        try:
            logger.info("Making prediction...")
            
            # Make prediction with memory optimization
            with tf.device('/cpu:0'):
                predictions = self.model.predict(processed_image, verbose=0, batch_size=1)
            
            # Get top prediction
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = self.class_names[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            # Get top 5 predictions
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            top_5_predictions = []
            
            for idx in top_5_indices:
                top_5_predictions.append({
                    'pest_name': self.class_names[idx],
                    'confidence': float(predictions[0][idx])
                })
            
            # Categorize by crop
            crop = self.categorize_pest_by_crop(predicted_class_name)
            
            # Clear memory after prediction
            del processed_image
            del predictions
            gc.collect()
            
            logger.info(f"Prediction completed: {predicted_class_name} ({confidence:.3f})")
            
            return {
                'predicted_pest': predicted_class_name,
                'confidence': confidence,
                'crop_category': crop,
                'top_5_predictions': top_5_predictions
            }, None
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, f"Prediction error: {e}"
    
    def categorize_pest_by_crop(self, pest_name):
        """Categorize pest by crop type"""
        pest_lower = pest_name.lower()
        
        if any(rice_pest in pest_lower for rice_pest in ["rice", "paddy", "asiatic", "brown plant", "small brown", "white backed", "yellow rice"]):
            return "Rice"
        elif any(wheat_pest in pest_lower for wheat_pest in ["wheat", "sawfly"]):
            return "Wheat"
        elif any(cotton_pest in pest_lower for cotton_pest in ["bollworm", "stem borer", "whitefly"]):
            return "Cotton"
        elif "thrips" in pest_lower:
            return "Sugarcane"
        else:
            return "Unknown"

# Initialize model manager
pest_manager = PestModelManager()

@app.route('/')
def index():
    """Homepage with API information"""
    return render_template('index.html')

@app.route('/api/pest/predict', methods=['POST'])
def predict_pest():
    """Predict pest type from uploaded image"""
    try:
        logger.info("Received prediction request")
        
        if not pest_manager.model:
            logger.error("Model not loaded")
            return jsonify({"error": "Pest detection model not loaded"}), 500
        
        if 'image' not in request.files:
            logger.error("No image file provided")
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        # Check file size (limit to 10MB)
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            logger.error(f"File too large: {file_size} bytes")
            return jsonify({"error": "File too large. Please upload an image smaller than 10MB."}), 400
        
        logger.info(f"Processing image: {image_file.filename} ({file_size} bytes)")
        
        result, error = pest_manager.predict_pest(image_file)
        
        if error:
            logger.error(f"Prediction failed: {error}")
            return jsonify({"error": error}), 500
        
        logger.info("Prediction successful")
        return jsonify({
            "success": True,
            "prediction": result
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/pest/classes')
def get_pest_classes():
    """Get list of all pest classes"""
    if not pest_manager.class_names:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Categorize classes by crop
    crop_categories = {
        "Rice": [],
        "Wheat": [],
        "Cotton": [],
        "Sugarcane": []
    }
    
    for pest_name in pest_manager.class_names:
        crop = pest_manager.categorize_pest_by_crop(pest_name)
        crop_categories[crop].append(pest_name)
    
    return jsonify({
        "total_classes": len(pest_manager.class_names),
        "crop_categories": crop_categories,
        "all_classes": pest_manager.class_names
    })

@app.route('/api/status')
def get_status():
    """Get API status and model information"""
    status = {
        "api_status": "running",
        "model_loaded": pest_manager.model is not None,
        "tensorflow_version": tf.__version__,
        "total_classes": len(pest_manager.class_names) if pest_manager.class_names else 0,
        "memory_usage": "optimized"
    }
    
    if pest_manager.class_names:
        status["supported_crops"] = ["Rice", "Wheat", "Cotton", "Sugarcane"]
        status["input_shape"] = pest_manager.input_shape
    
    return jsonify(status)

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": pest_manager.model is not None})

@app.route('/test')
def test_interface():
    """Test interface for pest detection"""
    return render_template('test_pest.html')

@app.route('/classes')
def classes_info():
    """Display pest classes information"""
    return render_template('classes_info.html')

if __name__ == '__main__':
    print("🐛 Pest Detection API Starting...")
    print("=" * 50)
    
    # Get port from environment variable (for deployment)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 API Endpoints:")
    print(f"   GET  /              - Homepage")
    print(f"   POST /api/pest/predict - Predict pest from image")
    print(f"   GET  /api/pest/classes  - Get pest classes")
    print(f"   GET  /api/status        - API status")
    print(f"   GET  /test             - Test interface")
    print(f"   GET  /classes          - Classes information")
    print(f"\n🚀 Starting Flask server on port {port}")
    print("=" * 50)
    
    app.run(debug=False, host='0.0.0.0', port=port)

