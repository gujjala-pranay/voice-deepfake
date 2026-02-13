@echo off
REM Training script for Audio Deepfake Detection Models using Kaggle Fake or Real Dataset
REM This script downloads the Fake or Real audio dataset and trains all models

setlocal enabledelayedexpansion

echo.
echo ===============================================================================
echo   Audio Deepfake Detection - Complete Training Pipeline
echo ===============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found

REM Check dependencies
echo.
echo Checking dependencies...
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch not found!
    echo Run: pip install torch torchvision librosa
    pause
    exit /b 1
)

python -c "import kagglehub; print('✅ kagglehub installed')" 2>nul
if errorlevel 1 (
    echo ⚠️  kagglehub not found!
    echo Run: pip install kagglehub
    echo Then set environment variable KAGGLE_API_TOKEN
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo   SETUP OPTIONS
echo ===============================================================================
echo.
echo 1. Train all models (Enhanced, Lightweight, Ensemble) - 50 epochs
echo 2. Train with custom epochs
echo 3. Train specific models only
echo 4. Download dataset only
echo 5. Skip download (use existing dataset)
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting training: All models, 50 epochs
    python train_all_models_kaggle.py ^
        --epochs 50 ^
        --batch_size 32 ^
        --models enhanced lightweight ensemble_standard
    goto end
)

if "%choice%"=="2" (
    set /p epochs="Enter number of epochs: "
    echo.
    echo 🚀 Starting training: All models, !epochs! epochs
    python train_all_models_kaggle.py ^
        --epochs !epochs! ^
        --batch_size 32 ^
        --models enhanced lightweight ensemble_standard
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Available models:
    echo   - enhanced
    echo   - lightweight
    echo   - ensemble_standard
    echo   - ensemble_multiscale
    echo   - ensemble_adaptive
    echo.
    set /p models="Enter models (space-separated): "
    echo.
    echo 🚀 Starting training: !models!
    python train_all_models_kaggle.py ^
        --epochs 50 ^
        --batch_size 32 ^
        --models !models!
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 📥 Downloading dataset only...
    python -c "from train_all_models_kaggle import KaggleTrainingPipeline; p = KaggleTrainingPipeline(); p.download_dataset()"
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 🚀 Starting training with existing dataset...
    python train_all_models_kaggle.py ^
        --epochs 50 ^
        --batch_size 32 ^
        --skip_download ^
        --models enhanced lightweight ensemble_standard
    goto end
)

echo Invalid choice
goto end

:end
echo.
echo ===============================================================================
echo   Training pipeline completed!
echo ===============================================================================
echo.
echo 💡 Next steps:
echo   1. View trained models in ./model/ directory
echo   2. Run: streamlit run app.py  (to test models)
echo.
pause
