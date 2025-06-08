#!/usr/bin/env python3
"""
Simplified YOLO11 Figure Detection Training
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = Path("../logs")
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

def train_model(model_name="../models/yolo11n.pt", data_config="../figure_dataset/data.yaml", 
                epochs=100, batch_size=16, img_size=640):
    """
    Train YOLO11 model with specified parameters
    
    Args:
        model_name: Path to YOLO11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        data_config: Path to dataset configuration file
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
    
    Available YOLO11 models:
    - yolo11n.pt: Nano (fastest, smallest)
    - yolo11s.pt: Small (good balance)
    - yolo11m.pt: Medium (higher accuracy)
    - yolo11l.pt: Large (even higher accuracy)
    - yolo11x.pt: Extra Large (highest accuracy, slowest)
    """
    
    logger = setup_logging()
    
    try:
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
            project="../models/train",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save=True,
            patience=15,
            save_period=10
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Copy best model to models root for easy access
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        models_root = Path("../models")
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

def train_with_model_size(model_size="n", epochs=100):
    """
    Train with specific YOLO11 model size
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
    """
    model_path = f"../models/yolo11{model_size}.pt"
    return train_model(model_name=model_path, epochs=epochs)

if __name__ == "__main__":
    # Default training with YOLO11n
    train_model()
    
    # Examples for other models:
    # train_with_model_size("s", epochs=50)  # YOLO11s
    # train_with_model_size("m", epochs=100) # YOLO11m
    # train_with_model_size("l", epochs=150) # YOLO11l
    # train_with_model_size("x", epochs=200) # YOLO11x 