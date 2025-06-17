#!/usr/bin/env python3
"""
Download all YOLO11 model variants
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_yolo11_models():
    """Download all YOLO11 model variants to the models directory"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # YOLO11 model variants
    models = {
        "yolo11n.pt": "YOLO11 Nano - Fastest, smallest (2.6M params)",
        "yolo11s.pt": "YOLO11 Small - Balanced speed/accuracy (9.4M params)", 
        "yolo11m.pt": "YOLO11 Medium - Higher accuracy (20.1M params)",
        "yolo11l.pt": "YOLO11 Large - Even higher accuracy (25.3M params)",
        "yolo11x.pt": "YOLO11 Extra Large - Best accuracy (56.9M params)"
    }
    
    print("🚀 Downloading YOLO11 models...")
    print("=" * 50)
    
    for model_name, description in models.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"✅ {model_name} already exists")
        else:
            print(f"⬇️  Downloading {model_name}...")
            print(f"   {description}")
            
            try:
                # This will download the model if it doesn't exist
                model = YOLO(model_name)
                
                # Move to models directory if not already there
                if not model_path.exists():
                    import shutil
                    shutil.move(model_name, model_path)
                
                print(f"✅ Downloaded {model_name}")
                
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
        
        print()
    
    print("🎉 YOLO11 models download complete!")
    print("\nAvailable models in models/ directory:")
    
    for model_file in sorted(models_dir.glob("yolo11*.pt")):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  {model_file.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    download_yolo11_models() 