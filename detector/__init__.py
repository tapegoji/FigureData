#!/usr/bin/env python3
"""
FigureDetect: A YOLOv5-based figure detection system for document images.

This package provides tools for detecting and extracting figures/diagrams from document images
using a pre-trained YOLOv5 model. It supports both single image processing and batch processing
of multiple images with optional recursive directory traversal.
"""

__version__ = "1.1.0"
__author__ = "innovoltive"
__email__ = "info@innovoltive.com"
__description__ = "A YOLOv5-based figure detection system for document images"

from .figure_detector import FigureDetector

__all__ = [
    "FigureDetector",
] 