#!/usr/bin/env python3
"""
Setup script for the figuredetect package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="figuredetect",
    version="1.1.0",
    author="Figure Detection Team",
    author_email="contact@figuredetect.com",
    description="A YOLOv5-based figure detection system for document images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/figuredetect",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "pandas>=1.1.4",
        "pyyaml>=5.3.1",
        "tqdm>=4.64.0",
        "seaborn>=0.11.0",
        "gitpython",
        "ipython",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "figuredetect=figuredetect.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "figuredetect": ["*.py"],
    },
    zip_safe=False,
) 