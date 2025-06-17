# Figure Detection Project

A YOLO11-based figure detection system that can automatically identify and extract figures/diagrams from documents and images using the latest Ultralytics YOLO architecture.

## Project Structure

```
FigureData/
├── detector/                   # Main detection modules
│   ├── __init__.py
│   ├── tra### Recent Updates

### Latest Implementation
- **Complete YOLO11 Integration**: Uses YOLO11 architecture exclusively
- **Streamlined Workflow**: Simple 3-step process (split → train → detect)
- **Dual Dataset Support**: Handles multiple datasets in `dataset/data/` and `dataset/old_data/`
- **Automated Data Splitting**: Smart train/validation splitting with `scripts/split_data.py`
- **Enhanced Training**: Simplified training script with comprehensive logging
- **Improved Detection**: Robust figure detection with visualization and cropping
- **Better Organization**: Clear project structure with dedicated directories for each component_detector.py  # Training script
│   └── figure_detector.py     # Detection script
├── scripts/                   # Utility scripts
│   ├── __init__.py
│   ├── split_data.py          # Data splitting script
│   ├── download_models.py     # Script to download YOLO11 models
│   └── test_gpu.py           # GPU testing utility
├── dataset/                   # Training dataset
│   ├── data.yaml             # Dataset configuration
│   ├── data/                 # Raw images and labels (all data)
│   ├── old_data/             # Backup of old data
│   ├── train/                # Training images and labels (after split)
│   └── val/                  # Validation images and labels (after split)
├── models/                    # Model files
│   ├── yolo11n.pt            # YOLO11 Nano (2.6M params, fastest)
│   ├── yolo11s.pt            # YOLO11 Small (9.4M params, balanced)
│   ├── yolo11m.pt            # YOLO11 Medium (20.1M params, higher accuracy)
│   ├── yolo11l.pt            # YOLO11 Large (25.3M params, even better)
│   ├── yolo11x.pt            # YOLO11 Extra Large (56.9M params, best)
│   ├── best.pt               # Best trained model (after training)
│   └── train/                # Training run outputs
├── input_images/              # Sample input images for testing
├── output/                    # Detection results and visualizations
├── logs/                      # Training and detection logs
├── yolo11n.pt                # Base YOLO11 model in root
├── pyproject.toml            # Project configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start Guide

Follow these 3 simple steps to get started with figure detection:

### Step 1: Split the Dataset

First, split your data from the `dataset/data` folder into training and validation sets:

```bash
python scripts/split_data.py
```

This will:
- Split your data into 70% training and 30% validation sets
- Create `dataset/train/` and `dataset/val/` directories
- Copy images to `images/` subdirectories and labels to `labels/` subdirectories
- Use the existing data in `dataset/data/` and `dataset/old_data/`

### Step 2: Train the Model

Train a YOLO11 model using your split dataset:

```bash
python detector/train_figure_detector.py
```

This will:
- Train a YOLO11n model for 100 epochs (default)
- Save training logs to `logs/training_YYYYMMDD_HHMMSS.log`
- Save the best model to `models/best.pt`
- Create detailed training outputs in `models/train/`

### Step 3: Run Figure Detection

Detect figures in your images:

```bash
python detector/figure_detector.py
```

This will:
- Use the trained model (`models/best.pt`) to detect figures
- Save detection results to `output/` directory
- Generate annotated visualizations and individual figure crops
- Create detection logs in `logs/detection_YYYYMMDD_HHMMSS.log`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FigureData
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
python scripts/download_models.py
```
This will download all YOLO11 variants (n, s, m, l, x) to the `models/` directory.

## Detailed Usage

### Data Preparation

The project includes two datasets in the `dataset/` folder:
- `dataset/data/` - Main dataset with images and annotations
- `dataset/old_data/` - Additional or backup dataset

Before training, you need to split your data:

```bash
python scripts/split_data.py
```

**Split Configuration:**
- Training ratio: 70% (modifiable in the script)
- Validation ratio: 30%
- Output: Creates `dataset/train/` and `dataset/val/` with proper YOLO structure

### Training a Model

To train a custom figure detection model:

```bash
python detector/train_figure_detector.py
```

**Training Parameters** (modifiable in the script):
- `model_name`: Path to YOLO11 model (default: "models/yolo11n.pt")
- `data_config`: Path to dataset config (default: "dataset/data.yaml") 
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Training batch size (default: 16)
- `img_size`: Input image size (default: 640)

**Training with Different YOLO11 Models:**
Edit the script to use different model sizes:
```python
# In train_figure_detector.py main section:
train_model("models/yolo11n.pt", epochs=100)  # YOLO11n - fastest
train_model("models/yolo11s.pt", epochs=120)  # YOLO11s - balanced
train_model("models/yolo11m.pt", epochs=100)  # YOLO11m - higher accuracy
train_model("models/yolo11l.pt", epochs=80)   # YOLO11l - even better
train_model("models/yolo11x.pt", epochs=60)   # YOLO11x - best accuracy
```

**Training Output:**
- Trained model saved to `models/train/run_YYYYMMDD_HHMMSS/`
- Best model automatically copied to `models/best.pt`
- Training logs saved to `logs/training_YYYYMMDD_HHMMSS.log`
- Training metrics and plots generated

### Running Detection

To detect figures in images:

```bash
python detector/figure_detector.py
```

**Detection Features:**
- Uses the best trained model (`models/best.pt`)
- Processes images from `input_images/` directory
- Configurable confidence threshold (default: 0.25)
- Saves results to `output/` directory

**Detection Output:**
- Annotated visualization with bounding boxes
- Individual figure crops saved as separate images
- Detection logs with confidence scores and coordinates
- Results organized by input image name

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

The training dataset follows YOLO format and includes:

### Data Structure
```
dataset/
├── data.yaml              # Dataset configuration file
├── data/                  # Raw dataset (all images and labels)
│   ├── *.jpg             # Image files
│   └── *.txt             # Corresponding label files
├── old_data/             # Additional/backup dataset
├── train/                # Training set (created by split_data.py)
│   ├── images/           # Training images
│   └── labels/           # Training labels
└── val/                  # Validation set (created by split_data.py)
    ├── images/           # Validation images
    └── labels/           # Validation labels
```

### Label Format
- Images in JPG/PNG format
- Labels in TXT format (one per image)
- Label format: `class_id x_center y_center width height` (normalized 0-1)
- Current dataset has 1 class: `diagram` (class_id = 0)

### Dataset Configuration (data.yaml)
```yaml
train: train/images
val: val/images
test: val/images

# Classes
nc: 1
names: ['diagram']
```

## Features

- **Data Splitting**: Automatically split datasets into training/validation sets
- **Training**: Train custom YOLO11 models for figure detection using the latest architecture
- **Detection**: Detect figures in images with confidence scores
- **Visualization**: Generate annotated images showing detected figures
- **Extraction**: Save individual detected figures as separate image files
- **Logging**: Comprehensive logging for all operations

## Logging

All operations are logged with timestamps:
- **Training logs**: `logs/training_YYYYMMDD_HHMMSS.log`
- **Detection logs**: `logs/detection_YYYYMMDD_HHMMSS.log`

Logs include:
- Process start/end times
- Model loading information
- Training metrics and progress
- Detection results and confidence scores
- Error messages and debugging information

## Performance

### Training Performance
- Training time: Varies by model size and epochs
- Memory usage: ~2GB GPU memory (recommended)
- Dataset: Split from combined datasets in `dataset/data/` and `dataset/old_data/`

### Detection Performance
- Inference speed: ~1-2ms per image (GPU)
- Confidence scores: Configurable threshold (default: 0.25)
- Supports batch processing of images

## Complete Workflow Example

Here's a complete example of the typical workflow:

```bash
# 1. Split your dataset
python scripts/split_data.py

# 2. Train the model
python detector/train_figure_detector.py

# 3. Run detection on your images
python detector/figure_detector.py
```

**Expected Output:**
- Split datasets in `dataset/train/` and `dataset/val/`
- Trained model saved as `models/best.pt`
- Detection results in `output/` directory
- Comprehensive logs in `logs/` directory

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

For questions or issues, please check the logs in `logs/` directory first, then create an issue in the repository. 