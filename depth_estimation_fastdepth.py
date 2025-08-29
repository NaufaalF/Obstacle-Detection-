import numpy as np
import cv2
import tensorflow as tf

class FastDepthEstimator:
    """
    FastDepth depth estimation using TensorFlow Lite
    Optimized for MobileNet-based architecture like original FastDepth
    """
    
    def __init__(self, model_path="./models/mobilenet_depth.tflite", input_size=(320, 320)):
        self.model_path = model_path
        self.input_size = input_size
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ FastDepth model loaded: {model_path}")
        print(f"📊 Input shape: {self.input_details[0]['shape']}")
        print(f"📊 Output shape: {self.output_details[0]['shape']}")
        print(f"📊 Input size: {input_size}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for FastDepth model
        Updated to support flexible input sizes including 320x320
        """
        # Resize to model input size (now supports 320x320)
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # ImageNet normalization (works well for 320x320 too)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def estimate_depth(self, image):
        """
        Estimate depth map using FastDepth model
        
        Args:
            image: Input image (numpy array, BGR format)
            
        Returns:
            depth_map: Depth map (numpy array), values in meters
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            depth_output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Remove batch dimension and get depth map
            depth_map = depth_output[0]
            
            # Handle different output formats
            if len(depth_map.shape) == 3 and depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]  # Remove channel dimension
            
            # Resize back to original image size
            original_h, original_w = image.shape[:2]
            depth_map = cv2.resize(depth_map, (original_w, original_h))
            
            # FastDepth postprocessing
            # Convert from model output to actual depth in meters
            # This depends on the specific FastDepth model used
            depth_map = self.postprocess_depth(depth_map)
            
            return depth_map
            
        except Exception as e:
            print(f"❌ Error in FastDepth estimation: {e}")
            # Return default depth map (assume 5 meter distance)
            return np.full((image.shape[0], image.shape[1]), 5.0, dtype=np.float32)
    
    def postprocess_depth(self, raw_depth):
        """
        Convert raw model output to actual depth values
        FastDepth typically outputs depth in different scales
        """
        # Method 1: If model outputs direct depth values
        if np.max(raw_depth) > 10:  # Likely in millimeters or different scale
            depth_map = raw_depth / 1000.0  # Convert mm to meters
            depth_map = np.clip(depth_map, 0.1, 10.0)
        
        # Method 2: If model outputs inverse depth
        elif np.max(raw_depth) <= 1.0:  
            # Inverse depth to actual depth
            depth_map = 1.0 / (raw_depth + 1e-6)
            depth_map = np.clip(depth_map, 0.1, 10.0)
        
        # Method 3: Linear scaling
        else:
            # Normalize to reasonable range
            depth_min, depth_max = np.min(raw_depth), np.max(raw_depth)
            if depth_max > depth_min:
                normalized = (raw_depth - depth_min) / (depth_max - depth_min)
                depth_map = 0.1 + normalized * 9.9  # Scale to 0.1-10 meters
            else:
                depth_map = np.full_like(raw_depth, 5.0)
        
        return depth_map.astype(np.float32)
    
    def get_object_depth(self, depth_map, bbox):
        """
        Get average depth of object from bounding box
        
        Args:
            depth_map: Depth map array
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            average_depth: Average depth in the bounding box area (meters)
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return 5.0  # Default 5 meters
        
        # Extract region of interest
        roi_depth = depth_map[y1:y2, x1:x2]
        
        # Calculate median depth (more robust than mean)
        # Filter out invalid values
        valid_depths = roi_depth[(roi_depth > 0.1) & (roi_depth < 10.0)]
        
        if len(valid_depths) > 0:
            median_depth = np.median(valid_depths)
        else:
            median_depth = 5.0  # Default fallback
        
        return float(median_depth)
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_PLASMA):
        """
        Convert depth map to colored visualization
        FastDepth style with better contrast
        
        Args:
            depth_map: Depth map array
            colormap: OpenCV colormap (PLASMA looks better for depth)
            
        Returns:
            colored_depth: Colored depth map for visualization
        """
        # Normalize depth map to 0-255 with better contrast
        depth_clipped = np.clip(depth_map, 0.1, 10.0)
        
        # Use logarithmic scaling for better visualization
        log_depth = np.log(depth_clipped + 1e-6)
        normalized = cv2.normalize(log_depth, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap (PLASMA gives better depth perception)
        colored = cv2.applyColorMap(normalized, colormap)
        
        return colored

# Alias for compatibility
DepthEstimator = FastDepthEstimator

# Test function
def test_fastdepth():
    """Test FastDepth model"""
    import os
    
    # Check for FastDepth model
    model_paths = [
        "models/mobilenet_depth.tflite",  # FastDepth equivalent
        "models/fastdepth.tflite",        # Converted FastDepth
        "models/midas_v21_small.tflite"   # Fallback
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No depth model found!")
        print("💡 Run: python get_fastdepth.py")
        return
    
    print(f"🎯 Testing FastDepth with: {model_path}")
    
    # Initialize FastDepth
    if "mobilenet" in model_path or "fastdepth" in model_path:
        depth_est = FastDepthEstimator(model_path, input_size=(320, 320))  # Updated to 320x320
    else:
        depth_est = FastDepthEstimator(model_path, input_size=(256, 256))  # MiDaS fallback
    
    # Test with camera or sample
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for testing
        frame = cv2.resize(frame, (640, 480))
        
        # Estimate depth
        depth_map = depth_est.estimate_depth(frame)
        
        # Visualize depth
        depth_colored = depth_est.visualize_depth(depth_map)
        
        # Show results
        cv2.imshow("Original", frame)
        cv2.imshow("FastDepth", depth_colored)
        
        # Print center depth
        center_depth = depth_map[240, 320]
        print(f"Center depth: {center_depth:.2f} meters", end='\r')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fastdepth()