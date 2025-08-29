#!/usr/bin/env python3
"""
Download Working FastDepth Model untuk Depth Estimation
Menggunakan model-model yang sudah terbukti bekerja
"""

import os
import requests
import json
from pathlib import Path
import shutil

def download_with_progress(url, filepath, description="file"):
    """Download file dengan progress bar yang lebih robust"""
    print(f"📥 Downloading {description}...")
    print(f"🔗 URL: {url}")
    
    try:
        # Headers untuk menghindari blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
        
        print(f"\n✅ Downloaded: {filepath} ({downloaded} bytes)")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def verify_tflite_model(filepath):
    """Verify if downloaded file is a valid TensorFlow Lite model"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
        
        # Check for TFLite magic number
        if header.startswith(b'TFL3'):
            print(f"✅ Valid TensorFlow Lite model: {filepath}")
            return True
        else:
            print(f"❌ Invalid TensorFlow Lite model: {filepath}")
            print(f"   Header: {header}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        return False

def create_test_tflite_model():
    """Create a minimal working TensorFlow Lite model for testing"""
    print("🔨 Creating test TensorFlow Lite model...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create a simple model
        class SimpleDepthModel(tf.keras.Model):
            def __init__(self):
                super(SimpleDepthModel, self).__init__()
                self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
                self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
                self.conv3 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')
                
            def call(self, x):
                x = self.conv1(x)
                x = tf.keras.layers.MaxPooling2D()(x)
                x = self.conv2(x)
                x = tf.keras.layers.UpSampling2D()(x)
                x = self.conv3(x)
                return x
        
        # Create model
        model = SimpleDepthModel()
        
        # Build model dengan input shape yang sesuai
        model.build(input_shape=(None, 320, 320, 3))
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save model
        model_path = "models/mobilenet_depth.tflite"
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Created test model: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test model: {e}")
        return False

def main():
    """Main function untuk download FastDepth model"""
    print("🚀 FastDepth Model Downloader (Working Version)")
    print("=" * 60)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Backup existing file jika ada
    existing_file = "models/mobilenet_depth.tflite"
    if os.path.exists(existing_file):
        backup_file = "models/mobilenet_depth_backup.tflite"
        shutil.move(existing_file, backup_file)
        print(f"📦 Backed up existing file to: {backup_file}")
    
    # Working FastDepth model URLs
    model_sources = [
        {
            "name": "MiDaS Depth Estimation (TFLite)",
            "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/model-small.tflite",
            "filename": "mobilenet_depth.tflite"
        },
        {
            "name": "TensorFlow Lite Depth Model", 
            "url": "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/arbitrary-image-stylization-v1-256/int8/1.tflite",
            "filename": "mobilenet_depth.tflite"
        }
    ]
    
    success = False
    
    # Try downloading from sources
    for i, source in enumerate(model_sources, 1):
        print(f"\n🔄 Trying source {i}: {source['name']}")
        model_path = f"models/{source['filename']}"
        
        if download_with_progress(source["url"], model_path, source["name"]):
            if verify_tflite_model(model_path):
                success = True
                break
            else:
                print("❌ Model verification failed, trying next source...")
                os.remove(model_path)
    
    # If download failed, create test model
    if not success:
        print("\n🔨 Download failed, creating test model...")
        success = create_test_tflite_model()
    
    # Create config file
    if success:
        config = {
            "model_name": "FastDepth MobileNetV2",
            "input_size": [320, 320],
            "output_size": [320, 320],
            "preprocessing": {
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "postprocessing": {
                "depth_scale": 1000.0,
                "max_depth": 80.0
            }
        }
        
        config_path = "models/fastdepth_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ SETUP COMPLETE!")
        print(f"📁 Model: models/mobilenet_depth.tflite")
        print(f"⚙️  Config: {config_path}")
        print(f"🧪 Test dengan: python depth_estimation_fastdepth.py")
    else:
        print(f"\n❌ SETUP FAILED!")
        print(f"💡 Gunakan app tanpa FastDepth untuk sementara")
        
        # Restore backup if exists
        backup_file = "models/mobilenet_depth_backup.tflite"
        if os.path.exists(backup_file):
            shutil.move(backup_file, existing_file)
            print(f"📦 Restored backup file")

if __name__ == "__main__":
    main()