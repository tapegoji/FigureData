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

class FigureDetector:
    """Simplified figure detector"""
    
    def __init__(self, model_path):
        self.logger = setup_logging()
        self.model = YOLO(model_path)
        self.logger.info(f"Loaded model: {model_path}")
    
    def detect_and_save(self, image_path, output_dir="output", conf_threshold=0.25):
        """Complete detection pipeline: detect, visualize, and save crops"""
        try:
            self.logger.info(f"Processing: {image_path}")
            
            # Run detection
            results = self.model.predict(source=image_path, conf=conf_threshold, verbose=False)[0]
            
            # Count detections
            count = len(results.boxes) if results.boxes is not None else 0
            self.logger.info(f"Found {count} figures")
            
            if count == 0:
                return {'count': 0, 'visualization': None, 'figures': []}
            
            # Create output directory structure
            image_path_obj = Path(image_path)
            parent_folder = image_path_obj.parent.name
            image_name = image_path_obj.stem
            image_output_dir = Path(output_dir) / parent_folder / image_name
            
            # Visualize and save
            visualization_path = image_output_dir / "detection_results.png"
            self.visualize(results, image_path, visualization_path)
            
            figures_dir = image_output_dir / "figures"
            saved_files = self.save_crops(results, image_path, figures_dir)
            
            return {
                'count': count,
                'visualization': visualization_path,
                'figures': saved_files
            }
            
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

if __name__ == "__main__":
    # Test with sample image or folder
    image_path = "input_images/Wolfspeed_C3M0032120K_data_sheet/"
    model_path = f"models/best.pt"
    conf_threshold = 0.7  # Confidence threshold for detection

    # Initialize detector
    detector = FigureDetector(model_path)

    # Check if path is a folder
    if Path(image_path).is_dir():
        # Find all valid images in folder and subfolders
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in valid_extensions:
            image_files.extend(Path(image_path).rglob(f"*{ext}"))
            image_files.extend(Path(image_path).rglob(f"*{ext.upper()}"))
        print(f"Found {len(image_files)} images to process")

        for img_file in image_files:
            print(f"Processing: {img_file}")
            results = detector.detect_and_save(str(img_file), conf_threshold=conf_threshold)
            if results:
                print(f"  Found {results['count']} figures")
    else:
        # Single image
        results = detector.detect_and_save(image_path, conf_threshold=conf_threshold)
        if results:
            print(f"Detection complete! Found {results['count']} figures.")
            print(f"Results: {results['visualization']}")
            print(f"Figures: {len(results['figures'])} saved") 