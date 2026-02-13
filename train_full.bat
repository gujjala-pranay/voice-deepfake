@echo off
echo ========================================
echo  Audio Deepfake Detection - Full Training
echo ========================================
echo.

set /p epochs="Enter number of epochs (default: 50): "
if "%epochs%"=="" set epochs=50

set /p batch_size="Enter batch size (default: 16): "
if "%batch_size%"=="" set batch_size=16

echo.
echo Configuration:
echo   Epochs: %epochs%
echo   Batch Size: %batch_size%
echo.

set TORCH_COMPILE_DISABLE=1
.venv\Scripts\python.exe train_elevenlabs.py --epochs %epochs% --batch_size %batch_size%

echo.
echo ========================================
echo  Training Complete!
echo ========================================
pause
