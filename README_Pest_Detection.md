# Pest Detection API - AgriMitra

🐛 **Advanced AI-powered pest detection for Rice, Wheat, Sugarcane, and Cotton crops**

## 🎯 Features

- **19 Pest Classes** across 4 major crops
- **64.4% Validation Accuracy** with efficient CNN architecture
- **Real-time Prediction** with confidence scores
- **Crop Categorization** for better pest management
- **Beautiful Web Interface** for easy testing
- **RESTful API** for integration

## 📊 Model Performance

- **Total Parameters**: 533,715
- **Model Size**: 2.04 MB
- **Training Images**: 28,500 (1,500 per pest class)
- **Validation Accuracy**: 64.4%
- **Supported Crops**: Rice (12 pests), Wheat (4 pests), Cotton (3 pests), Sugarcane (1 pest)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_pest.txt
```

### 2. Start the API

```bash
python pest_app.py
```

### 3. Access Web Interface

- **Homepage**: http://localhost:5000/
- **Test Interface**: http://localhost:5000/test
- **Classes Info**: http://localhost:5000/classes

## 🔧 API Endpoints

### POST /api/pest/predict

Upload an image to get pest predictions.

**Request:**

```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/api/pest/predict
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "predicted_pest": "asiatic rice borer",
    "confidence": 0.8542,
    "crop_category": "Rice",
    "top_5_predictions": [
      {"pest_name": "asiatic rice borer", "confidence": 0.8542},
      {"pest_name": "rice leaf caterpillar", "confidence": 0.0891},
      ...
    ]
  }
}
```

### GET /api/pest/classes

Get all supported pest classes organized by crop.

**Response:**

```json
{
  "total_classes": 19,
  "crop_categories": {
    "Rice": ["asiatic rice borer", "brown plant hopper", ...],
    "Wheat": ["sawfly", "wheat blossom midge", ...],
    "Cotton": ["bollworm", "stem borer", "whitefly"],
    "Sugarcane": ["Thrips"]
  }
}
```

### GET /api/status

Check API status and model information.

**Response:**

```json
{
  "api_status": "running",
  "model_loaded": true,
  "total_classes": 19,
  "supported_crops": ["Rice", "Wheat", "Cotton", "Sugarcane"],
  "tensorflow_version": "2.10.0"
}
```

## 🌾 Supported Pest Classes

### Rice Pests (12 types)

- asiatic rice borer
- brown plant hopper
- paddy stem maggot
- rice gall midge
- rice leaf caterpillar
- rice leaf roller
- rice leafhopper
- rice shell pest
- Rice Stemfly
- rice water weevil
- small brown plant hopper
- white backed plant hopper
- yellow rice borer

### Wheat Pests (4 types)

- sawfly
- wheat blossom midge
- wheat phloeothrips
- wheat sawfly

### Cotton Pests (3 types)

- bollworm
- stem borer
- whitefly

### Sugarcane Pests (1 type)

- Thrips

## 🏗️ Model Architecture

```
Model: "filtered_pest_detector"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
input_image (InputLayer)    [(None, 224, 224, 3)]     0
conv2d (Conv2D)             (None, 224, 224, 32)      896
batch_normalization (BatchN (None, 224, 224, 32)     128
max_pooling2d (MaxPooling2D (None, 112, 112, 32)     0
dropout (Dropout)           (None, 112, 112, 32)      0
conv2d_1 (Conv2D)           (None, 112, 112, 64)      18496
batch_normalization_1 (Batc (None, 112, 112, 64)     256
max_pooling2d_1 (MaxPooling (None, 56, 56, 64)       0
dropout_1 (Dropout)         (None, 56, 56, 64)        0
conv2d_2 (Conv2D)           (None, 56, 56, 128)       73856
batch_normalization_2 (Batc (None, 56, 56, 128)      512
max_pooling2d_2 (MaxPooling (None, 28, 28, 128)      0
dropout_2 (Dropout)         (None, 28, 28, 128)      0
conv2d_3 (Conv2D)           (None, 28, 28, 256)       295168
batch_normalization_3 (Batc (None, 28, 28, 256)      1024
global_average_pooling2d (G (None, 256)              0
dropout_3 (Dropout)         (None, 256)               0
dense (Dense)               (None, 512)               131584
batch_normalization_4 (Batc (None, 512)              2048
dropout_4 (Dropout)         (None, 512)               0
output (Dense)              (None, 19)                9747
=================================================================
Total params: 533,715
Trainable params: 531,731
Non-trainable params: 1,984
```

## 🧪 Testing

Run the test script to verify API functionality:

```bash
python test_api.py
```

## 📁 File Structure

```
├── pest_app.py                    # Flask API application
├── templates/                     # HTML templates
│   ├── index.html                # Homepage
│   ├── test_pest.html            # Test interface
│   └── classes_info.html         # Classes information
├── filtered_pest_model_best.h5   # Trained model
├── filtered_pest_model_info.json # Model metadata
├── requirements_pest.txt         # Python dependencies
├── test_api.py                   # API testing script
└── README_Pest_Detection.md      # This file
```

## 🔧 Technical Details

- **Framework**: TensorFlow 2.10.0
- **Architecture**: Efficient CNN with BatchNorm and Dropout
- **Input Size**: 224x224x3 RGB images
- **Optimization**: Adam optimizer with learning rate scheduling
- **Data Augmentation**: Rotation, flip, zoom, brightness, contrast
- **GPU Support**: RTX 4060 compatible

## 🌐 Web Interface Features

### Homepage (/)

- API overview and statistics
- Endpoint documentation
- Quick access to test interface

### Test Interface (/test)

- Drag & drop image upload
- Real-time prediction display
- Top 5 predictions with confidence scores
- Crop categorization
- Image preview

### Classes Information (/classes)

- Complete pest class listings
- Organized by crop type
- Interactive interface

## 🚀 Deployment

For production deployment, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 pest_app:app
```

## 📝 Notes

- The model was trained on balanced dataset with 1,500 images per pest class
- GPU acceleration is supported for faster inference
- All images are automatically resized to 224x224 pixels
- Model automatically categorizes pests by crop type
- Confidence scores help assess prediction reliability

## 🤝 Contributing

This is part of the AgriMitra project for agricultural pest detection and management.

---

**🎉 Happy Pest Detection!** 🐛🌾

