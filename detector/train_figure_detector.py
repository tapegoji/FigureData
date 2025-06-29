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

def train_model(model_name="models/yolo11m.pt", data_config="dataset/data.yaml", 
                epochs=100, batch_size=16, img_size=640):
    """
    Train YOLO11 model with specified parameters
    
    Args:
        model_name: Path to YOLO11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        data_config: Path to dataset configuration file
        epochs: Number of training epochs
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
    # Default training with YOLO11m
    # Uses existing model from models/ folder
    # train_model("models/yolo11n.pt", epochs=100)
    
    # Examples for other models:
    # train_model("models/yolo11n.pt", epochs=150)  # YOLO11n
    train_model("models/yolo11s.pt", epochs=120)  # YOLO11s
    # train_model("models/yolo11l.pt", epochs=80)   # YOLO11l
    # train_model("models/yolo11x.pt", epochs=60)   # YOLO11x 