import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from PIL import Image
import os

class TFLiteConverter:
    """Convert and optimize TensorFlow models to TensorFlow Lite format"""
    
    def __init__(self, model_path, class_names=None):
        self.model_path = model_path
        self.model = None
        self.tflite_model = None
        self.interpreter = None
        self.class_names = class_names or []
        
    def load_keras_model(self):
        """Load the Keras model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return self.model
    
    def convert_to_tflite(self, output_path='model.tflite', optimization='default'):
        """
        Convert Keras model to TensorFlow Lite
        
        Args:
            output_path: Path to save the TFLite model
            optimization: 'default', 'dynamic_range', 'float16', 'int8'
        """
        if self.model is None:
            self.load_keras_model()
        
        print(f"\nConverting model with {optimization} optimization...")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimization based on type
        if optimization == 'default':
            # No optimization
            converter.optimizations = []
            
        elif optimization == 'dynamic_range':
            # Dynamic range quantization (weights to int8)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif optimization == 'float16':
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif optimization == 'int8':
            # Full integer quantization (requires representative dataset)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # Convert the model
        self.tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(self.tflite_model)
        
        # Get model size
        model_size = os.path.getsize(output_path) / 1024  # KB
        print(f"✓ Model converted and saved to {output_path}")
        print(f"✓ Model size: {model_size:.2f} KB")
        
        return output_path
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for int8 quantization"""
        # This is a placeholder - you should provide real data
        for _ in range(100):
            # Generate random data matching your input shape
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    def load_tflite_model(self, tflite_path):
        """Load TensorFlow Lite model"""
        print(f"\nLoading TFLite model from {tflite_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        print("✓ TFLite model loaded!")
        return self.interpreter
    
    def get_model_details(self):
        """Get input and output details of TFLite model"""
        if self.interpreter is None:
            raise ValueError("TFLite model not loaded!")
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        print("\n=== Model Details ===")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        return input_details, output_details
    
    def preprocess_image(self, image_path, input_details):
        """Preprocess image for TFLite model"""
        # Get input shape and type
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((input_shape[1], input_shape[2]))
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize based on model type
        if input_dtype == np.uint8:
            # For quantized models
            img_array = img_array.astype(np.uint8)
        else:
            # For float models - MobileNetV2 preprocessing
            img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_tflite(self, image_path):
        """Make prediction using TFLite model"""
        if self.interpreter is None:
            raise ValueError("TFLite model not loaded!")
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Preprocess image
        input_data = self.preprocess_image(image_path, input_details)
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        
        # Handle quantized output
        if output_details[0]['dtype'] == np.uint8:
            # Dequantize
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
        
        # Get prediction
        predictions = output_data[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        return predicted_class, confidence, inference_time


class TFLiteTester:
    """Test TFLite model on sample dataset"""
    
    def __init__(self, converter):
        self.converter = converter
        self.results = []
    
    def test_on_dataset(self, test_dir):
        """Test model on all images in directory"""
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"Error: Directory {test_dir} does not exist!")
            return None
        
        print(f"\n=== Testing on dataset: {test_dir} ===")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in test_path.rglob('*') if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No images found in directory!")
            return None
        
        print(f"Found {len(image_files)} images")
        
        # Test each image
        inference_times = []
        predictions_list = []
        
        for i, image_path in enumerate(image_files, 1):
            try:
                predicted_class, confidence, inference_time = self.converter.predict_tflite(str(image_path))
                
                class_name = (self.converter.class_names[predicted_class] 
                             if predicted_class < len(self.converter.class_names) 
                             else f"Class_{predicted_class}")
                
                result = {
                    'image': image_path.name,
                    'predicted_class': int(predicted_class),
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'inference_time_ms': inference_time
                }
                
                self.results.append(result)
                inference_times.append(inference_time)
                predictions_list.append(predicted_class)
                
                print(f"[{i}/{len(image_files)}] {image_path.name}: "
                      f"{class_name} ({confidence:.2%}) - {inference_time:.2f}ms")
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
        
        # Calculate statistics
        if inference_times:
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            
            print(f"\n=== Performance Statistics ===")
            print(f"Average inference time: {avg_time:.2f} ms")
            print(f"Std deviation: {std_time:.2f} ms")
            print(f"Min time: {min_time:.2f} ms")
            print(f"Max time: {max_time:.2f} ms")
            print(f"FPS (theoretical): {1000/avg_time:.2f}")
        
        return self.results
    
    def save_results(self, output_path='test_results.json'):
        """Save test results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    def visualize_predictions(self, test_dir, num_samples=6):
        """Visualize predictions on sample images"""
        test_path = Path(test_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in test_path.rglob('*') if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No images found!")
            return
        
        # Select random samples
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Create visualization
        rows = (num_samples + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, image_path in enumerate(samples):
            if idx >= num_samples:
                break
            
            # Get prediction
            predicted_class, confidence, inference_time = self.converter.predict_tflite(str(image_path))
            class_name = (self.converter.class_names[predicted_class] 
                         if predicted_class < len(self.converter.class_names) 
                         else f"Class_{predicted_class}")
            
            # Load and display image
            img = Image.open(image_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(
                f"{class_name}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Time: {inference_time:.2f}ms",
                fontsize=10
            )
        
        # Hide extra subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved to predictions_visualization.png")
        plt.show()
    
    def compare_models(self, models_info):
        """Compare different model optimizations"""
        print("\n=== Model Comparison ===")
        print(f"{'Model':<20} {'Size (KB)':<12} {'Avg Time (ms)':<15} {'FPS':<10}")
        print("-" * 60)
        
        for model_name, size, avg_time in models_info:
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"{model_name:<20} {size:<12.2f} {avg_time:<15.2f} {fps:<10.2f}")


# Main execution function
def main():
    """Main function to convert and test TFLite model"""
    
    # Configuration
    KERAS_MODEL_PATH = 'recyclable_classifier.h5'
    TEST_DATA_DIR = 'test_images/'
    CLASS_NAMES = ['trash_bags', 'recyclable_plastic', 'recyclable_paper']  # Update with your classes
    
    print("="*60)
    print("TensorFlow Lite Model Converter and Tester")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"Error: Model file {KERAS_MODEL_PATH} not found!")
        print("Please train the model first using the training script.")
        return
    
    # Initialize converter
    converter = TFLiteConverter(KERAS_MODEL_PATH, CLASS_NAMES)
    converter.load_keras_model()
    
    # Convert to different formats
    models_info = []
    
    # 1. Default (no optimization)
    print("\n" + "="*60)
    print("1. Converting to TFLite (Default - No Optimization)")
    print("="*60)
    tflite_path = converter.convert_to_tflite('model_default.tflite', 'default')
    default_size = os.path.getsize(tflite_path) / 1024
    
    # 2. Dynamic range quantization
    print("\n" + "="*60)
    print("2. Converting to TFLite (Dynamic Range Quantization)")
    print("="*60)
    tflite_dr_path = converter.convert_to_tflite('model_dynamic_range.tflite', 'dynamic_range')
    dr_size = os.path.getsize(tflite_dr_path) / 1024
    
    # 3. Float16 quantization
    print("\n" + "="*60)
    print("3. Converting to TFLite (Float16 Quantization)")
    print("="*60)
    tflite_f16_path = converter.convert_to_tflite('model_float16.tflite', 'float16')
    f16_size = os.path.getsize(tflite_f16_path) / 1024
    
    # Test the dynamic range model (good balance)
    print("\n" + "="*60)
    print("Testing Dynamic Range Quantized Model")
    print("="*60)
    converter.load_tflite_model(tflite_dr_path)
    converter.get_model_details()
    
    # Test on dataset
    if os.path.exists(TEST_DATA_DIR):
        tester = TFLiteTester(converter)
        results = tester.test_on_dataset(TEST_DATA_DIR)
        
        if results:
            tester.save_results('tflite_test_results.json')
            tester.visualize_predictions(TEST_DATA_DIR, num_samples=6)
            
            # Get average time for comparison
            avg_time = np.mean([r['inference_time_ms'] for r in results])
            models_info.append(('Dynamic Range', dr_size, avg_time))
    else:
        print(f"\nTest directory '{TEST_DATA_DIR}' not found.")
        print("Creating sample prediction...")
        
        # Create a dummy test if no test data available
        print("\nTo test the model, create a directory structure like:")
        print(f"  {TEST_DATA_DIR}")
        print("    ├── image1.jpg")
        print("    ├── image2.jpg")
        print("    └── ...")
    
    # Model comparison
    models_info.extend([
        ('Default (No Opt)', default_size, 0),
        ('Float16', f16_size, 0)
    ])
    
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)
    for name, size, _ in models_info:
        print(f"{name:<20}: {size:>10.2f} KB")
    
    print("\n✓ Conversion and testing complete!")
    print(f"\nGenerated files:")
    print(f"  - model_default.tflite")
    print(f"  - model_dynamic_range.tflite (Recommended)")
    print(f"  - model_float16.tflite")
    if os.path.exists(TEST_DATA_DIR):
        print(f"  - tflite_test_results.json")
        print(f"  - predictions_visualization.png")


if __name__ == "__main__":
    main()
