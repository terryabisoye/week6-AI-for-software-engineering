# Edge AI & Smart Agriculture Examples

This repository contains sample projects and utilities for Edge AI workflows, model conversion to TensorFlow Lite, and a Smart Agriculture UI prototype.

Contents
- [analysis.txt](analysis.txt) — theoretical notes about Edge AI and related comparisons.
- [example.py](example.py) — MobileNetV2-based image classifier implementation (class: `RecyclableItemsClassifier`).
- [example2.py](example2.py) — TensorFlow Lite conversion & testing utilities (`TFLiteConverter`, `TFLiteTester`).
- [AI diven concept/example3.py](AI diven concept/example3.py) — React UI prototype for a Smart Agriculture System (`SmartAgricultureSystem`).

Quick start

1. Prepare environment
   - Python 3.8+ recommended.
   - Install dependencies:
     ```
     pip install tensorflow numpy matplotlib pillow
     ```

2. Train or obtain a Keras model
   - Use the model code in [example.py](example.py) to create and save a model:
     - Create model: use `RecyclableItemsClassifier.create_model()`
     - Train on your dataset and save as `recyclable_classifier.h5`

3. Convert to TFLite and test
   - Use [example2.py](example2.py):
     - `TFLiteConverter('<path-to-h5>', class_names=...)` loads and converts models.
     - Supported optimizations: `default`, `dynamic_range`, `float16`, `int8`.
     - `TFLiteTester` runs inference across images in a folder and saves results/visualizations.

   Example (from repo root):
   ```
   python example2.py
   ```
   - The script expects `recyclable_classifier.h5` and a `test_images/` folder (see the top of [example2.py](example2.py) for configuration).
   - Outputs: `model_default.tflite`, `model_dynamic_range.tflite`, `model_float16.tflite`, and (if tests run) `tflite_test_results.json` and `predictions_visualization.png`.

Notes & tips
- Representative dataset: for full integer quantization (`int8`) provide a realistic representative dataset generator in `TFLiteConverter._representative_dataset_gen()` for best accuracy.
- Preprocessing: `TFLiteConverter.preprocess_image()` assumes MobileNet-style float preprocessing or uint8 for quantized models. Adjust if your model uses different preprocessing.
- Model-to-TFLite conversion of non-Keras models or SavedModel formats will require using the appropriate `tf.lite.TFLiteConverter` constructor.

Smart Agriculture UI
- The React component [AI diven concept/example3.py](AI diven concept/example3.py) (exported as `SmartAgricultureSystem`) is a UI prototype demonstrating sensors, AI models, data flow, and implementation guidance. Integrate into a React project (file appears to be JSX/TSX content in a .py-named file — rename to `.jsx`/`.tsx` if you want to run it in a frontend project).

References
- See model and tester implementations: [`TFLiteConverter`](example2.py), [`TFLiteTester`](example2.py)
- See classifier scaffold: [`RecyclableItemsClassifier`](example.py)
- See UI prototype: [`SmartAgricultureSystem`](AI diven concept/example3.py)
- Analysis notes: [analysis.txt](analysis.txt)

License / Usage
- Use these scripts as examples / starting points. Validate and adapt preprocessing, quantization representative data, and deployment steps to your production requirements.
