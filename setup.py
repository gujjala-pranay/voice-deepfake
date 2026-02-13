from setuptools import setup, find_packages

setup(
    name="audio-deepfake-detection",
    version="1.0.0",
    description="Audio Deepfake Detection using 2D CNN",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/audio-deepfake-detection",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "streamlit>=1.28.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu121",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deepfake audio detection CNN machine-learning",
)
