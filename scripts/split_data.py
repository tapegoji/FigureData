#!/usr/bin/env python3
"""
Split data into training and validation sets for YOLO training
"""

import os
import shutil
import random
from pathlib import Path

def split_data(data_dir, train_dir, val_dir, train_ratio=0.7):
    """
    Split data into training and validation sets
    
    Args:
        data_dir: Source directory containing .jpg and .txt files
        train_dir: Destination directory for training data
        val_dir: Destination directory for validation data
        train_ratio: Ratio of data to use for training (0.8 = 80% train, 20% val)
    """
    
    # Create output directories
    train_images_dir = Path(train_dir) / "images"
    train_labels_dir = Path(train_dir) / "labels"
    val_images_dir = Path(val_dir) / "images"
    val_labels_dir = Path(val_dir) / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all .jpg files
    data_path = Path(data_dir)
    image_files = list(data_path.glob("*.jpg"))
    
    print(f"Found {len(image_files)} image files")
    
    # Filter to only include images that have corresponding .txt files
    valid_pairs = []
    for img_file in image_files:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            valid_pairs.append((img_file, txt_file))
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    # Shuffle the data for random split
    random.seed(42)  # For reproducible results
    random.shuffle(valid_pairs)
    
    # Calculate split point
    split_point = int(len(valid_pairs) * train_ratio)
    
    train_pairs = valid_pairs[:split_point]
    val_pairs = valid_pairs[split_point:]
    
    print(f"Training set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")
    
    # Copy training data
    for img_file, txt_file in train_pairs:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(txt_file, train_labels_dir / txt_file.name)
    
    # Copy validation data
    for img_file, txt_file in val_pairs:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(txt_file, val_labels_dir / txt_file.name)
    
    print("Data split completed successfully!")
    print(f"Training data saved to: {train_dir}")
    print(f"Validation data saved to: {val_dir}")

if __name__ == "__main__":
    # Define paths
    data_dir = "/home/asepahvand/repos/FigureData/dataset/data"
    train_dir = "/home/asepahvand/repos/FigureData/dataset/train"
    val_dir = "/home/asepahvand/repos/FigureData/dataset/validation"
    
    # Remove existing directories if they exist
    for directory in [train_dir, val_dir]:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"Removing existing directory: {directory}")
            shutil.rmtree(dir_path)

    # Split the data (70% train, 30% validation)
    split_data(data_dir, train_dir, val_dir, train_ratio=0.7)
