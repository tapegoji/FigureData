#!/usr/bin/env python3
"""
Simplified YOLO11 Figure Detection Training
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import yaml

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_optimal_epochs(train_images, validation_images, model_size="n"):
    """
    Calculate optimal number of epochs based on dataset size and model complexity
    
    Args:
        train_images: Number of training images
        validation_images: Number of validation images
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
    
    Returns:
        Optimal number of epochs
    """
    total_images = train_images + validation_images
    
    # Base epochs calculation
    # For small datasets (< 500 images), use more epochs to ensure good learning
    # For larger datasets, fewer epochs are needed
    
    if total_images < 100:
        base_epochs = 300
    elif total_images < 200:
        base_epochs = 200
    elif total_images < 500:
        base_epochs = 150
    elif total_images < 1000:
        base_epochs = 100
    else:
        base_epochs = 80
    
    # Adjust based on model size (larger models need fewer epochs to avoid overfitting)
    model_multipliers = {
        'n': 1.2,  # Nano needs more epochs
        's': 1.1,  # Small needs slightly more
        'm': 1.0,  # Medium is baseline
        'l': 0.9,  # Large needs fewer
        'x': 0.8   # Extra large needs even fewer
    }
    
    epochs = int(base_epochs * model_multipliers.get(model_size, 1.0))
    
    # Ensure minimum and maximum bounds
    epochs = max(100, min(epochs, 500))
    
    return epochs

def count_dataset_images(data_config="figure_dataset/data.yaml"):
    """
    Count the number of training and validation images in the dataset
    
    Args:
        data_config: Path to dataset configuration file
    
    Returns:
        Tuple of (train_count, val_count)
    """
    import yaml
    import os
    
    # Read the data config
    with open(data_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the base directory of the config file
    config_dir = Path(data_config).parent
    
    # Count training images
    train_path = config_dir / config['train']
    train_count = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png'))) + len(list(train_path.glob('*.jpeg')))
    
    # Count validation images
    val_path = config_dir / config['val']
    val_count = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png'))) + len(list(val_path.glob('*.jpeg')))
    
    return train_count, val_count

def train_model(model_name="models/yolo11m.pt", data_config="figure_dataset/data.yaml", 
                epochs=None, batch_size=16, img_size=640):
    """
    Train YOLO11 model with specified parameters
    
    Args:
        model_name: Path to YOLO11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        data_config: Path to dataset configuration file
        epochs: Number of training epochs (if None, will be calculated automatically)
        batch_size: Training batch size
        img_size: Input image size
    """
    
    logger = setup_logging()
    
    try:
        # Check if model exists
        model_path = Path(model_name)
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}. Please run the download script first.")
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        logger.info(f"Using model: {model_path}")
        
        # Auto-calculate epochs if requested
        if epochs is None:
            train_count, val_count = count_dataset_images(data_config)
            # Extract model size from model name
            model_size = model_name.split('yolo11')[-1].split('.')[0] if 'yolo11' in model_name else 'n'
            calculated_epochs = calculate_optimal_epochs(train_count, val_count, model_size)
            
            if epochs is None:
                epochs = calculated_epochs
                logger.info(f"Auto-calculated epochs: {epochs} (based on {train_count} training + {val_count} validation images)")
            else:
                logger.info(f"Using specified epochs: {epochs} (recommended: {calculated_epochs})")
        
        logger.info("Starting YOLO11 training")
        logger.info(f"Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}")
        
        # Load model
        model = YOLO(model_name)
        
        # Train
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project="models/train",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save=True,
            patience=15,
            save_period=10
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Copy best model to models root for easy access
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        models_root = Path("models")
        models_root.mkdir(exist_ok=True)
        
        if best_model_path.exists():
            import shutil
            dest_path = models_root / "best.pt"
            shutil.copy2(best_model_path, dest_path)
            logger.info(f"Best model copied to: {dest_path}")
        else:
            logger.warning("Best model not found in expected location")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Default training with YOLO11m - epochs will be auto-calculated
    # Uses existing model from models/ folder
    train_model("models/yolo11m.pt")
    
    # Examples for other models with auto-calculated epochs:
    # train_model("models/yolo11n.pt")  # YOLO11n with 
    # train_model("models/yolo11s.pt")  # YOLO11s with 
    # train_model("models/yolo11l.pt")  # YOLO11l with 
    # train_model("models/yolo11x.pt")  # YOLO11x with 

    # Examples with manual epochs:
    # train_model("models/yolo11s.pt", epochs=150, auto_epochs=False)  # YOLO11s with manual epochs
    # train_model("models/yolo11m.pt", epochs=120, auto_epochs=False)  # YOLO11m with manual epochs 