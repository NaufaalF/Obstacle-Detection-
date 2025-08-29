#!/usr/bin/env python3
"""
Simple test untuk FastDepth setup
Test semua komponen sebelum menjalankan aplikasi utama
"""
import os
import sys
import numpy as np

def test_model_files():
    """Test apakah semua model file ada"""
    print("🔍 Checking model files...")
    
    files_to_check = [
        ("models/suBest3_float16.tflite", "YOLO Model"),
        ("models/mobilenet_depth.tflite", "FastDepth Model"), 
        ("models/fastdepth_config.json", "FastDepth Config")
    ]
    
    all_exist = True
    total_size = 0
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            total_size += size_mb
            print(f"✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {description}: {file_path} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print(f"📊 Total model size: {total_size:.1f} MB")
    
    return all_exist

def test_python_packages():
    """Test apakah semua package Python terinstall"""
    print("\n🔍 Testing Python packages...")
    
    required_packages = [
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"), 
        ("flask", "Flask"),
        ("flask_socketio", "Flask-SocketIO"),
        ("ultralytics", "Ultralytics YOLO")
    ]
    
    all_installed = True
    
    for package_name, description in required_packages:
        try:
            if package_name == "cv2":
                import cv2
                version = cv2.__version__
            elif package_name == "tensorflow":
                import tensorflow as tf
                version = tf.__version__
            elif package_name == "numpy":
                import numpy as np
                version = np.__version__
            elif package_name == "flask":
                import flask
                version = flask.__version__
            elif package_name == "ultralytics":
                import ultralytics
                version = "installed"
            
            print(f"✅ {description}: v{version}")
            
        except ImportError:
            print(f"❌ {description}: NOT INSTALLED")
            all_installed = False
            
    return all_installed

def test_fastdepth_model():
    """Test FastDepth model loading dan inference"""
    print("\n🔍 Testing FastDepth model...")
    
    try:
        # Check if FastDepth implementation exists
        if not os.path.exists("depth_estimation_fastdepth.py"):
            print("❌ depth_estimation_fastdepth.py not found")
            print("💡 Make sure to copy the FastDepth implementation file")
            return False
        
        # Import FastDepth
        sys.path.insert(0, '.')
        from depth_estimation_fastdepth import FastDepthEstimator
        
        # Find model
        model_path = None
        possible_paths = [
            "models/mobilenet_depth.tflite",
            "models/fastdepth.tflite"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ No FastDepth model found")
            return False
        
        print(f"📁 Using model: {model_path}")
        
        # Initialize FastDepth with 320x320
        depth_estimator = FastDepthEstimator(model_path, input_size=(320, 320))
        print("✅ FastDepth model loaded successfully")
        
        # Test with dummy image
        print("🧪 Testing inference with dummy image...")
        dummy_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        depth_map = depth_estimator.estimate_depth(dummy_image)
        
        print(f"✅ Inference successful!")
        print(f"   Input shape: {dummy_image.shape}")
        print(f"   Output shape: {depth_map.shape}")
        print(f"   Depth range: {depth_map.min():.2f} - {depth_map.max():.2f} meters")
        
        # Test object depth extraction
        bbox = [100, 100, 200, 200]  # dummy bbox
        object_depth = depth_estimator.get_object_depth(depth_map, bbox)
        print(f"   Object depth test: {object_depth:.2f} meters")
        
        return True
        
    except Exception as e:
        print(f"❌ FastDepth test failed: {e}")
        print("💡 Error details:")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🔍 Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        yolo_path = "models/suBest3_float16.tflite"
        if not os.path.exists(yolo_path):
            print(f"❌ YOLO model not found: {yolo_path}")
            return False
        
        print(f"📁 Loading YOLO: {yolo_path}")
        model = YOLO(yolo_path)
        print("✅ YOLO model loaded successfully")
        
        # Test prediction with dummy image
        print("🧪 Testing YOLO inference...")
        dummy_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        results = model.predict(dummy_image, imgsz=320, conf=0.25, verbose=False)
        
        print("✅ YOLO inference successful")
        print(f"   Results type: {type(results)}")
        print(f"   Number of results: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def test_camera_access():
    """Test camera access"""
    print("\n🔍 Testing camera access...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera (not available or permission denied)")
            print("💡 Check:")
            print("   - Camera is connected")
            print("   - No other app is using camera") 
            print("   - Camera permissions granted")
            return False
        
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera working!")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Frame type: {frame.dtype}")
        else:
            print("❌ Cannot read frame from camera")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_flask_components():
    """Test Flask dan SocketIO"""
    print("\n🔍 Testing Flask components...")
    
    try:
        from flask import Flask
        from flask_socketio import SocketIO
        
        # Create test app
        app = Flask(__name__)
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
        
        print("✅ Flask application created")
        print("✅ SocketIO initialized")
        
        # Test route
        @app.route("/test")
        def test_route():
            return "OK"
        
        print("✅ Test route added")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FastDepth Setup Test\n")
    print("="*60)
    
    tests = [
        ("Model Files", test_model_files),
        ("Python Packages", test_python_packages), 
        ("FastDepth Model", test_fastdepth_model),
        ("YOLO Model", test_yolo_model),
        ("Camera Access", test_camera_access),
        ("Flask Components", test_flask_components)
    ]
    
    results = []
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name.upper()}")
        print("-" * 40)
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            passed += 1
            print(f"🎯 {test_name}: PASSED")
        else:
            print(f"💥 {test_name}: FAILED")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✨ Your FastDepth setup is ready!")
        print("\n🚀 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Click 'Start Camera' and test!")
        
    elif passed >= 4:  # Core components working
        print(f"\n⚠️  PARTIAL SUCCESS ({passed}/{len(tests)} passed)")
        print("🔧 Core components are working, you can try running the app")
        print("💡 Fix the failing tests for optimal performance")
        
    else:
        print(f"\n❌ TOO MANY FAILURES ({len(tests)-passed} failed)")
        print("🛠️  Please fix the issues before running the app:")
        
        # Show specific fixes
        for test_name, result in results:
            if not result:
                if "Model Files" in test_name:
                    print("   💡 Run: python get_fastdepth.py")
                elif "Python Packages" in test_name:
                    print("   💡 Run: pip install tensorflow opencv-python ultralytics flask flask-socketio")
                elif "FastDepth Model" in test_name:
                    print("   💡 Make sure depth_estimation_fastdepth.py exists")
                elif "Camera Access" in test_name:
                    print("   💡 Check camera connection and permissions")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)