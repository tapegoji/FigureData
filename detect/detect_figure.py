#!/usr/bin/env python3
"""
Simplified YOLO11 Figure Detection
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

class SimpleDetector:
    """Simplified figure detector"""
    
    def __init__(self, model_path):
        self.logger = setup_logging()
        self.model = YOLO(model_path)
        self.logger.info(f"Loaded model: {model_path}")
    
    def detect(self, image_path, conf_threshold=0.25):
        """Detect figures in image"""
        try:
            self.logger.info(f"Processing: {image_path}")
            
            # Run detection
            results = self.model.predict(source=image_path, conf=conf_threshold, verbose=False)
            
            # Count detections
            count = len(results[0].boxes) if results[0].boxes is not None else 0
            self.logger.info(f"Found {count} figures")
            
            return results[0]
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return None
    
    def visualize(self, results, image_path, output_path="output/detection.png"):
        """Visualize detection results"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create plot
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image)
            
            # Draw boxes
            if results.boxes is not None:
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Draw rectangle
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x1, y1-10, f"Figure: {conf:.2f}", 
                           fontsize=10, color='red', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(f"Detected {len(results.boxes) if results.boxes else 0} Figures", 
                        fontsize=14, weight='bold')
            ax.axis('off')
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved: {output_path}")
            return len(results.boxes) if results.boxes else 0
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return 0
    
    def save_crops(self, results, image_path, output_dir="output/figures"):
        """Save detected figures as individual images"""
        try:
            if results.boxes is None or len(results.boxes) == 0:
                self.logger.warning("No figures to crop")
                return []
            
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # Crop figure
                crop = image[y1:y2, x1:x2]
                
                # Save
                filename = f"figure_{i+1}_conf_{conf:.3f}.png"
                filepath = Path(output_dir) / filename
                
                from PIL import Image
                Image.fromarray(crop).save(filepath)
                saved_files.append(str(filepath))
                
                self.logger.info(f"Saved: {filename} (conf: {conf:.1%})")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Cropping failed: {e}")
            return []

def detect_figures(image_path, model_path=None, output_dir="output"):
    """Main detection function"""
    
    # Find model
    if model_path is None:
        # First check for best model in models root
        best_model = Path("models/best.pt")
        if best_model.exists():
            model_path = best_model
        else:
            # Look for trained model in subdirectories
            train_models = list(Path("models/train").glob("*/weights/best.pt"))
            if train_models:
                model_path = train_models[0]
            else:
                # Try to find any YOLO11 model, prefer larger models for better accuracy
                for model_size in ['x', 'l', 'm', 's', 'n']:
                    yolo11_model = Path(f"models/yolo11{model_size}.pt")
                    if yolo11_model.exists():
                        model_path = yolo11_model
                        break
                else:
                    model_path = "yolo11n.pt"  # Download if none found
    
    # Setup detector
    detector = SimpleDetector(model_path)
    
    # Run detection
    results = detector.detect(image_path)
    if results is None:
        return
    
    # Visualize
    output_path = Path(output_dir) / "detection_results.png"
    count = detector.visualize(results, image_path, output_path)
    
    # Save crops
    figures_dir = Path(output_dir) / "figures"
    saved_files = detector.save_crops(results, image_path, figures_dir)
    
    return {
        'count': count,
        'visualization': output_path,
        'figures': saved_files
    }

def detect_with_model_size(image_path, model_size="n", output_dir="output"):
    """
    Detect figures using specific YOLO11 model size
    
    Args:
        image_path: Path to input image
        model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
        output_dir: Output directory for results
    
    Returns:
        Detection results dictionary
    """
    model_path = f"models/yolo11{model_size}.pt"
    return detect_figures(image_path, model_path=model_path, output_dir=output_dir)

if __name__ == "__main__":
    # Test with sample image using best available model
    image_path = "data/Wolfspeed_C3M0032120K_data_sheet/page_5.png"
    results = detect_figures(image_path)
    
    if results:
        print(f"Detection complete! Found {results['count']} figures.")
        print(f"Results: {results['visualization']}")
        print(f"Figures: {len(results['figures'])} saved")
        
    # Examples for specific models:
    # results_m = detect_with_model_size(image_path, "m")  # Use YOLO11m
    # results_l = detect_with_model_size(image_path, "l")  # Use YOLO11l 