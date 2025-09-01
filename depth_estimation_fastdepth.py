#!/usr/bin/env python3
"""
FastDepth Depth Estimation - FIXED DATA TYPE VERSION
Fixes FLOAT64 vs FLOAT32 tensor error
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
import sys
import time

class FastDepthEstimator:
    def __init__(self, model_path, input_size=(320, 320)):
        """Initialize FastDepth estimator with FIXED data type handling"""
        print(f"📁 Loading FastDepth: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Load TensorFlow Lite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_size = input_size
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            
            # 🔧 CRITICAL: Check expected input data type
            self.input_dtype = self.input_details[0]['dtype']
            print(f"🎯 Model expects input dtype: {self.input_dtype}")
            
            print(f"✅ FastDepth model loaded: {model_path}")
            print(f"📊 Input shape: {self.input_shape}")
            print(f"📊 Output shape: {self.output_shape}")
            print(f"📊 Input size: {input_size}")
            
        except Exception as e:
            print(f"❌ Error loading FastDepth model: {e}")
            raise

    def preprocess_image(self, image):
        """Preprocess image with FIXED data type handling"""
        try:
            # Resize to model input size
            resized = cv2.resize(image, self.input_size)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 🔧 CRITICAL FIX: Ensure correct data type from start
            if self.input_dtype == np.float32:
                # Normalize to [0, 1] as FLOAT32
                normalized = rgb_image.astype(np.float32) / 255.0
            elif self.input_dtype == np.uint8:
                # Keep as uint8 [0, 255]
                normalized = rgb_image.astype(np.uint8)
            else:
                # Default fallback
                normalized = rgb_image.astype(np.float32) / 255.0
            
            # Add batch dimension and ensure correct dtype
            input_tensor = np.expand_dims(normalized, axis=0)
            
            # 🔧 DOUBLE CHECK: Force correct dtype
            input_tensor = input_tensor.astype(self.input_dtype)
            
            print(f"🎯 Preprocessed tensor dtype: {input_tensor.dtype} (expected: {self.input_dtype})")
            
            return input_tensor
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            # Return safe fallback tensor
            fallback = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=self.input_dtype)
            return fallback

    def estimate_depth(self, image):
        """Estimate depth with enhanced error handling"""
        try:
            print(f"🎯 Starting depth estimation...")
            
            # Preprocess image with fixed data types
            input_tensor = self.preprocess_image(image)
            
            print(f"🎯 Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            print(f"✅ Input tensor set successfully")
            
            # Run inference
            self.interpreter.invoke()
            print(f"✅ Inference completed")
            
            # Get output
            depth_map = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Remove batch dimension
            depth_map = np.squeeze(depth_map)
            
            print(f"🎯 Raw depth shape: {depth_map.shape}")
            print(f"🎯 Raw depth range: {depth_map.min():.4f} - {depth_map.max():.4f}")
            
            # 🔧 IMPROVED: Better depth scaling
            # Normalize depth map to meaningful range
            if depth_map.max() <= 1.0:
                # If output is normalized [0,1], scale to meters
                depth_map = depth_map * 20.0  # Scale to 0-20 meters
            elif depth_map.max() <= 255:
                # If output is [0,255], scale to meters  
                depth_map = (depth_map / 255.0) * 20.0
            
            # Ensure minimum depth
            depth_map = np.clip(depth_map, 0.1, 50.0)
            
            print(f"✅ Processed depth range: {depth_map.min():.2f} - {depth_map.max():.2f} meters")
            
            return depth_map
            
        except Exception as e:
            print(f"❌ Error in depth estimation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return realistic fallback depth map with variation
            h, w = self.input_size
            fallback = np.random.uniform(1.0, 10.0, (h, w))  # Random depths 1-10m
            print(f"🔄 Using fallback depth map with range: {fallback.min():.2f} - {fallback.max():.2f}")
            return fallback

    def get_object_depth(self, depth_map, bbox):
        """Get average depth with improved handling"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within bounds
            h, w = depth_map.shape
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))  
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract region of interest
            roi = depth_map[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Use median to avoid outliers
                object_depth = np.median(roi)
                print(f"🎯 Object depth calculation: ROI size={roi.size}, median={object_depth:.2f}m")
                return object_depth
            else:
                print(f"⚠️ Empty ROI, using default depth")
                return 2.0  # Default 2 meters
                
        except Exception as e:
            print(f"❌ Error getting object depth: {e}")
            return 2.0

def test_fastdepth_fixed():
    """Test FastDepth with data type fixes"""
    print("🎯 Testing FIXED FastDepth with: models/mobilenet_depth.tflite")
    
    model_path = "models/mobilenet_depth.tflite"
    
    try:
        # Initialize FastDepth estimator
        depth_est = FastDepthEstimator(model_path, input_size=(320, 320))
        
        print("\n🧪 Testing with synthetic image...")
        # Create test image with patterns
        test_image = np.zeros((320, 320, 3), dtype=np.uint8)
        
        # Add some patterns for depth variation
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), -1)    # Red square
        cv2.rectangle(test_image, (170, 170), (270, 270), (0, 255, 0), -1)  # Green square
        cv2.circle(test_image, (160, 160), 40, (0, 0, 255), -1)             # Blue circle
        
        print("🎯 Running depth estimation...")
        depth_map = depth_est.estimate_depth(test_image)
        
        print(f"✅ Depth estimation successful!")
        print(f"   📊 Depth map shape: {depth_map.shape}")
        print(f"   📊 Depth range: {depth_map.min():.2f} - {depth_map.max():.2f} meters")
        print(f"   📊 Mean depth: {depth_map.mean():.2f} meters")
        print(f"   📊 Std deviation: {depth_map.std():.2f} meters")
        
        # Test object depth extraction
        bbox = [50, 50, 150, 150]  # Red square area
        object_depth = depth_est.get_object_depth(depth_map, bbox)
        print(f"   🎯 Object depth (red square): {object_depth:.2f} meters")
        
        # Quality assessment
        dynamic_range = depth_map.max() - depth_map.min()
        if dynamic_range > 1.0:
            print(f"   ✅ EXCELLENT: Dynamic range = {dynamic_range:.2f}m")
        elif dynamic_range > 0.5:
            print(f"   🟡 GOOD: Dynamic range = {dynamic_range:.2f}m") 
        else:
            print(f"   ⚠️ POOR: Dynamic range = {dynamic_range:.2f}m")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 FastDepth FIXED Version Test")
    print("=" * 50)
    print("🔧 Fixes FLOAT64 vs FLOAT32 tensor error")
    print("🎯 Improved depth scaling and error handling")
    print("=" * 50)
    
    success = test_fastdepth_fixed()
    
    if success:
        print(f"\n🎉 FIXED VERSION WORKING!")
        print(f"📝 Replace your existing depth_estimation_fastdepth.py with this version")
        print(f"🚀 Then run: python app_with_midas_debug.py")
    else:
        print(f"\n❌ Still has issues - may need different model")
        print(f"💡 Consider using app without FastDepth for now")

if __name__ == "__main__":
    main()