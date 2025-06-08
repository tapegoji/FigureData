# Figure Detection Project

A YOLO11-based figure detection system that can automatically identify and extract figures/diagrams from documents and images using the latest Ultralytics YOLO architecture.

## Project Structure

```
figuredetect/
├── detect/                     # Main detection modules
│   ├── train_figure.py        # Training script
│   └── detect_figure.py       # Detection script
├── logs/                      # Training and detection logs
├── output/                    # Detection results and visualizations
├── download_models.py         # Script to download all YOLO11 models
├── models/                    # Model files
│   ├── yolo11n.pt            # YOLO11 Nano (2.6M params, fastest)
│   ├── yolo11s.pt            # YOLO11 Small (9.4M params, balanced)
│   ├── yolo11m.pt            # YOLO11 Medium (20.1M params, higher accuracy)
│   ├── yolo11l.pt            # YOLO11 Large (25.3M params, even better)
│   ├── yolo11x.pt            # YOLO11 Extra Large (56.9M params, best)
│   ├── best.pt               # Best trained model (after training)
│   └── train/                # Training run outputs
├── figure_dataset/            # Training dataset
│   ├── data.yaml             # Dataset configuration
│   ├── train/                # Training images and labels
│   └── validation/           # Validation images and labels
├── data/                     # Sample test data
└── requirements.txt          # Python dependencies
```

## Features

- **Training**: Train custom YOLO11 models for figure detection using the latest architecture
- **Detection**: Detect figures in images with confidence scores
- **Visualization**: Generate annotated images showing detected figures
- **Extraction**: Save individual detected figures as separate image files
- **Logging**: Comprehensive logging for both training and detection processes

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd figuredetect
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO11 models (optional - models auto-download when needed):
```bash
python download_models.py
```
This will download all YOLO11 variants (n, s, m, l, x) to the `models/` directory.

## Usage

### Training a Model

To train a custom figure detection model:

```bash
cd detect
python train_figure.py
```

**Training Parameters** (can be modified in the script):
- `model_name`: Path to YOLO11 model (default: "../models/yolo11n.pt")
- `data_config`: Path to dataset config (default: "../figure_dataset/data.yaml") 
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Training batch size (default: 16)
- `img_size`: Input image size (default: 640)

**Training with Different YOLO11 Models**:
```python
# In Python
from detect.train_figure import train_with_model_size

# Train with different model sizes
train_with_model_size("n", epochs=50)   # YOLO11n - fastest
train_with_model_size("s", epochs=75)   # YOLO11s - balanced
train_with_model_size("m", epochs=100)  # YOLO11m - higher accuracy
train_with_model_size("l", epochs=150)  # YOLO11l - even better
train_with_model_size("x", epochs=200)  # YOLO11x - best accuracy
```

**Training Output:**
- Trained model saved to `models/train/run_YYYYMMDD_HHMMSS/`
- Best model automatically copied to `models/best.pt`
- Training logs saved to `detect/logs/training_YYYYMMDD_HHMMSS.log`
- Training metrics and plots generated

### Running Detection

To detect figures in images:

```bash
cd detect
python detect_figure.py
```

**Detection Parameters** (can be modified in the script):
- `image_path`: Path to input image
- `model_path`: Path to YOLO11 model (auto-detects best available)
- `output_dir`: Output directory for results (default: "output")
- `conf_threshold`: Confidence threshold for detections (default: 0.25)

**Detection with Different YOLO11 Models**:
```python
# In Python
from detect.detect_figure import detect_with_model_size

# Detect with different model sizes
results = detect_with_model_size("image.jpg", "n")  # YOLO11n - fastest
results = detect_with_model_size("image.jpg", "m")  # YOLO11m - balanced
results = detect_with_model_size("image.jpg", "x")  # YOLO11x - best accuracy
```

**Detection Output:**
- Annotated visualization: `output/detection_results.png`
- Individual figure crops: `output/figures/figure_N_conf_X.XXX.png`
- Detection logs: `detect/logs/detection_YYYYMMDD_HHMMSS.log`

### Using as Library

You can also import and use the detection functions in your own code:

```python
from detect.detect_figure import detect_figures, detect_with_model_size
from detect.train_figure import train_with_model_size

# Detect figures in an image (auto-selects best available YOLO11 model)
results = detect_figures("path/to/image.png", output_dir="my_output")

# Or specify a particular YOLO11 model
results = detect_with_model_size("path/to/image.png", "m", "my_output")

# Train with specific YOLO11 model
train_results = train_with_model_size("l", epochs=100)

print(f"Found {results['count']} figures")
print(f"Visualization: {results['visualization']}")
print(f"Figure files: {results['figures']}")
```

## Model Information

### YOLO11 Models

The project exclusively uses YOLO11, the latest and most advanced YOLO architecture (2024):

| Model | Size | Parameters | mAP50-95 | Speed | Use Case |
|-------|------|------------|----------|-------|----------|
| **yolo11n.pt** | 5.6MB | 2.6M | 39.5% | Fastest | Real-time, edge devices |
| **yolo11s.pt** | 19MB | 9.4M | 47.0% | Fast | Balanced speed/accuracy |
| **yolo11m.pt** | 40MB | 20.1M | 51.5% | Medium | Higher accuracy needs |
| **yolo11l.pt** | 50MB | 25.3M | 53.4% | Slower | Professional applications |
| **yolo11x.pt** | 113MB | 56.9M | 54.7% | Slowest | Maximum accuracy |

**YOLO11 Architecture Features**:
- **Enhanced feature extraction** with improved backbone architecture
- **C3k2 and C2PSA modules** for better efficiency and accuracy
- **Optimized training pipeline** with early stopping
- **Better parameter efficiency** - higher accuracy with fewer parameters
- **Future-proof design** - latest computer vision innovations

### Training Results

After training, you can expect:
- High accuracy figure detection (typically >95% mAP50)
- Real-time inference speeds
- Robust detection across various figure types (charts, diagrams, plots, etc.)

## Dataset Format

The training dataset should follow YOLO format:
- Images in JPG/PNG format
- Labels in TXT format (one per image)
- Label format: `class_id x_center y_center width height` (normalized 0-1)
- `data.yaml` configuration file specifying paths and classes

## Logging

All operations are logged with timestamps:
- **Training logs**: `detect/logs/training_YYYYMMDD_HHMMSS.log`
- **Detection logs**: `detect/logs/detection_YYYYMMDD_HHMMSS.log`

Logs include:
- Process start/end times
- Model loading information
- Training metrics and progress
- Detection results and confidence scores
- Error messages and debugging information

## Performance

### Training Performance
- Training time: ~1-2 minutes for 10 epochs (GPU recommended)
- Memory usage: ~2GB GPU memory
- Dataset: 280 training images, 60 validation images

### Detection Performance
- Inference speed: ~1-2ms per image (GPU)
- Accuracy: >95% mAP50 on validation set
- Confidence scores: Typically 90%+ for clear figures

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure CUDA is properly installed for GPU acceleration
2. **Path Issues**: Make sure to run scripts from the correct directory
3. **Model Not Found**: Ensure training completed successfully and `best.pt` exists
4. **Low Detection Accuracy**: Consider training for more epochs or with more data

### Requirements
- Python 3.8+
- PyTorch with CUDA support (recommended)
- Sufficient disk space for model files and outputs
- GPU recommended for training (CPU works but slower)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Recent Updates

### Pure YOLO11 Implementation (Latest)
- **Exclusively YOLO11**: Completely migrated to YOLO11 architecture, removed all legacy code
- **Multi-Model Support**: Support for all YOLO11 variants (n, s, m, l, x)
- **Smart Model Selection**: Automatic detection of best available YOLO11 model
- **Enhanced Training Options**: Easy functions to train with different model sizes
- **Flexible Detection**: Choose model size based on speed vs accuracy needs
- **Future-Proof Architecture**: Using only the latest 2024 YOLO technology

---

For questions or issues, please check the logs in `detect/logs/` first, then create an issue in the repository. 