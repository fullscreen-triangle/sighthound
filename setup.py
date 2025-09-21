#!/usr/bin/env python3
"""
Setup script for the Sighthound Ultimate Validation Engine.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README_ENGINE.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = """
    Sighthound Ultimate Validation Engine
    
    The most comprehensive experimental validation framework for biometric-geolocation
    correlation analysis using revolutionary path reconstruction methodology.
    """

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "joblib>=1.0.0",
        "h5py>=3.0.0",
        "openpyxl>=3.0.0"
    ]

setup(
    name="sighthound-validation-engine",
    version="1.0.0",
    author="Sighthound Validation Team",
    author_email="validation@sighthound.ai",
    description="Revolutionary experimental validation framework for biometric-geolocation correlation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sighthound/validation-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0", 
            "flake8>=3.9.0",
            "mypy>=0.900"
        ],
        "gpu": [
            "tensorflow>=2.8.0",
            "torch>=1.10.0"
        ],
        "advanced": [
            "pykalman>=0.9.5",
            "sympy>=1.8",
            "networkx>=2.6"
        ]
    },
    entry_points={
        "console_scripts": [
            "sighthound-validate=demo_comprehensive_validation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "biometric", "geolocation", "validation", "consciousness", 
        "path-reconstruction", "virtual-spectroscopy", "atmospheric-modeling",
        "olympic-data", "experimental-validation", "machine-learning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/sighthound/validation-engine/issues",
        "Source": "https://github.com/sighthound/validation-engine",
        "Documentation": "https://sighthound-validation.readthedocs.io/",
    },
)