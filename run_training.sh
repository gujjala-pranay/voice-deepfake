#!/bin/bash

# Training script for Audio Deepfake Detection Models using Kaggle Fake or Real Dataset
# This script downloads the dataset and trains all models

set -e

echo ""
echo "==============================================================================="
echo "  Audio Deepfake Detection - Complete Training Pipeline"
echo "==============================================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check dependencies
echo ""
echo "Checking dependencies..."

if python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null; then
    :
else
    echo "❌ PyTorch not found!"
    echo "Run: pip3 install torch torchvision librosa"
    exit 1
fi

if python3 -c "import kagglehub; print('✅ kagglehub installed')" 2>/dev/null; then
    :
else
    echo "⚠️  kagglehub not found!"
    echo "Run: pip3 install kagglehub"
    echo "Then set: export KAGGLE_API_TOKEN=your_token_here"
    exit 1
fi

echo ""
echo "==============================================================================="
echo "  SETUP OPTIONS"
echo "==============================================================================="
echo ""
echo "1. Train all models (Enhanced, Lightweight, Ensemble) - 50 epochs"
echo "2. Train with custom epochs"
echo "3. Train specific models only"
echo "4. Download dataset only"
echo "5. Skip download (use existing dataset)"
echo ""

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting training: All models, 50 epochs"
        python3 train_all_models_kaggle.py \
            --epochs 50 \
            --batch_size 32 \
            --models enhanced lightweight ensemble_standard
        ;;
    2)
        echo ""
        read -p "Enter number of epochs: " epochs
        echo ""
        echo "🚀 Starting training: All models, $epochs epochs"
        python3 train_all_models_kaggle.py \
            --epochs "$epochs" \
            --batch_size 32 \
            --models enhanced lightweight ensemble_standard
        ;;
    3)
        echo ""
        echo "Available models:"
        echo "  - enhanced"
        echo "  - lightweight"
        echo "  - ensemble_standard"
        echo "  - ensemble_multiscale"
        echo "  - ensemble_adaptive"
        echo ""
        read -p "Enter models (space-separated): " models
        echo ""
        echo "🚀 Starting training: $models"
        python3 train_all_models_kaggle.py \
            --epochs 50 \
            --batch_size 32 \
            --models $models
        ;;
    4)
        echo ""
        echo "📥 Downloading dataset only..."
        python3 -c "from train_all_models_kaggle import KaggleTrainingPipeline; p = KaggleTrainingPipeline(); p.download_dataset()"
        ;;
    5)
        echo ""
        echo "🚀 Starting training with existing dataset..."
        python3 train_all_models_kaggle.py \
            --epochs 50 \
            --batch_size 32 \
            --skip_download \
            --models enhanced lightweight ensemble_standard
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "==============================================================================="
echo "  Training pipeline completed!"
echo "==============================================================================="
echo ""
echo "💡 Next steps:"
echo "   1. View trained models in ./model/ directory"
echo "   2. Run: streamlit run app.py  (to test models)"
echo ""
