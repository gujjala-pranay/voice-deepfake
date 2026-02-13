@echo off
chcp 65001 >nul
echo ========================================
echo  Audio Deepfake Detection - Training
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please run setup first: setup.bat
    pause
    exit /b 1
)

REM Set environment variable to disable torch.compile (Windows compatibility)
set TORCH_COMPILE_DISABLE=1

echo Starting training...
echo Model: Enhanced (ElevenLabs Dataset)
echo Epochs: 5 (quick test)
echo.

.venv\Scripts\python.exe train_simple.py

echo.
echo ========================================
echo  Training Complete!
echo ========================================
echo.
echo Trained model saved to: model\enhanced_elevenlabs.pth
echo.
echo Next steps:
echo   1. Run: streamlit run app.py
echo   2. Upload audio file to test
echo.
pause
