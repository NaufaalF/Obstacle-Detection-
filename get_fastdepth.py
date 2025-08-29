#!/usr/bin/env python3
"""
Download FastDepth TensorFlow Lite model that's ready to use
Based on MobileNet architecture similar to original FastDepth
"""
import os
import urllib.request
import ssl

def create_ssl_context():
    """Create SSL context for downloads"""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context

def download_file(url, filename, description=""):
    """Download file with progress"""
    print(f"📥 Downloading {description}...")
    print(f"    URL: {url}")
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r   Progress: {percent}%", end='', flush=True)
        
        # Create custom opener with SSL context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=create_ssl_context()))
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded: {filename}")
        
        # Check file size
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False

def get_fastdepth_alternatives():
    """Get FastDepth or similar lightweight depth models"""
    os.makedirs("models", exist_ok=True)
    
    print("🎯 FastDepth TensorFlow Lite Options:\n")
    
    # Option 1: MobileNetV2-based depth model (closest to FastDepth)
    print("Option 1: MobileNetV2 Depth Estimation (FastDepth-like)")
    mobilenet_depth_url = "https://tfhub.dev/google/lite-model/depth_estimation/1?lite-format=tflite"
    mobilenet_path = "models/mobilenet_depth.tflite"
    
    success1 = download_file(mobilenet_depth_url, mobilenet_path, "MobileNetV2 Depth Model")
    
    # Option 2: Create a FastDepth-equivalent model specification
    if success1:
        print(f"\n✅ FastDepth-equivalent model ready!")
        print(f"   Model: {mobilenet_path}")
        print(f"   Architecture: MobileNetV2-based (similar to FastDepth)")
        print(f"   Input: 224x224 RGB")
        print(f"   Output: Depth map")
        
        # Test the model
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=mobilenet_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"\n📊 Model Specifications:")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            print(f"   Input type: {input_details[0]['dtype']}")
            print(f"   Output type: {output_details[0]['dtype']}")
            
        except Exception as e:
            print(f"⚠️  Model validation failed: {e}")
    
    # Option 3: Manual instructions for true FastDepth
    print(f"\n" + "="*60)
    print("🔧 Alternative: Build True FastDepth from Source")
    print("="*60)
    print("If you really want the exact FastDepth model:")
    print("1. Run: python convert_fastdepth.py")
    print("2. This will convert the PyTorch model you already have")
    print("3. Requires: pip install onnx onnx-tf torch")
    
    return success1

def create_fastdepth_config():
    """Create configuration for FastDepth-like model"""
    config = {
        "model_name": "FastDepth-equivalent", 
        "architecture": "MobileNetV2 + Depth Head",
        "input_size": [320, 320],  # Updated to 320x320 for consistency
        "input_format": "RGB",
        "output_format": "depth_map",
        "preprocessing": {
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],  # ImageNet standard
            "std": [0.229, 0.224, 0.225]
        },
        "postprocessing": {
            "depth_scale": 1000,  # Convert to meters
            "min_depth": 0.1,
            "max_depth": 10.0
        }
    }
    
    import json
    with open("models/fastdepth_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ FastDepth configuration saved: models/fastdepth_config.json")
    return config

def main():
    print("🚀 FastDepth TensorFlow Lite Setup\n")
    
    # Get FastDepth alternatives
    success = get_fastdepth_alternatives()
    
    if success:
        # Create config
        create_fastdepth_config()
        
        print(f"\n🎉 FastDepth-equivalent setup complete!")
        print("\nNext steps:")
        print("1. Update your depth_estimation.py:")
        print("   - Change model path to 'models/mobilenet_depth.tflite'")
        print("   - Update input_size to (224, 224)")
        print("2. Test with: python simple_test.py")
        print("3. Run app: python app.py")
        
        # Show how to update depth_estimation.py
        print(f"\n💡 Update your depth_estimation.py:")
        print(f"   depth_estimator = DepthEstimator(")
        print(f'       "models/mobilenet_depth.tflite",')
        print(f'       input_size=(224, 224)')
        print(f"   )")
    else:
        print(f"\n❌ Download failed")
        print(f"💡 Alternative options:")
        print(f"1. Use MiDaS instead: python download_models.py")
        print(f"2. Convert manually: python convert_fastdepth.py") 
        print(f"3. Download manually from TensorFlow Hub")

if __name__ == "__main__":
    main()