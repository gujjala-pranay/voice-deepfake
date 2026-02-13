# 🎵 Audio Deepfake Detection System

A state-of-the-art deep learning system to detect synthetic audio, voice conversion, and deepfakes using advanced 2D CNN architectures with multiple ensemble strategies.

## 🎯 Overview

This system analyzes audio files to distinguish between:
- **REAL** (Bona fide): Genuine human speech
- **FAKE** (Deepfake/Synthetic): AI-generated or voice-converted speech

It uses multi-feature audio processing (Mel-Spectrograms, MFCC, Phase) combined with ensemble learning for maximum accuracy.

---

## ✨ Key Features

### 🎨 Multiple Model Architectures
- **Enhanced CNN**: Full-featured model using all 3 audio features
- **Lightweight CNN**: Fast real-time model for edge deployment
- **Ensemble Methods**:
  - Standard (Weighted averaging)
  - MultiScale (Multi-resolution processing)
  - Adaptive (Input-aware fusion)

### 🖼️ Advanced Audio Features
- **Mel-Spectrograms**: Multi-scale spectral analysis
- **MFCC**: Cepstral coefficients with delta features
- **Phase Features**: Instantaneous frequency analysis

### 🚀 Dual Interface
- **Streamlit Web App** (`app.py`): Interactive UI with visualizations
- **FastAPI REST API** (`main.py`): Production-ready backend

### 🎓 Comprehensive Dataset
- Trained on ASVspoof 2019 dataset
- Handles various synthetic speech and voice conversion techniques

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize Models
```bash
python init_models.py
```

### Step 3: Launch Streamlit App
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to use the app!

---

## 📖 Full Setup Guide

For detailed installation and configuration instructions, see [SETUP.md](SETUP.md)

---

## 📊 Model Comparison

| Model | Features | Speed | Accuracy | Best For |
|-------|----------|-------|----------|----------|
| Enhanced | 3 | Medium | High | General use |
| Lightweight | 1 | Very Fast | Medium | Real-time |
| Standard Ensemble | 3+ | Slow | Very High | High accuracy |
| MultiScale Ensemble | 3+ | Slowest | Very High | Complex patterns |
| Adaptive Ensemble | 3+ | Slow | Very High | Variable quality |

---

## 🖥️ Streamlit App

### Features
- **Upload Audio**: WAV or MP3 files
- **Real-time Analysis**: Waveform and spectrogram visualization
- **Model Selection**: Switch between different models
- **Confidence Scores**: Probability of prediction
- **Technical Explanations**: Understanding the model's reasoning

### Select Models in Sidebar
1. **Enhanced**: High accuracy with all features
2. **Lightweight**: Fast inference
3. **Ensemble**: Best performance with three strategies

---

## 🔌 FastAPI REST API

### Quick Test
```bash
python main.py
```

### Endpoints

#### Get Information
```bash
curl http://localhost:8000/
```

#### List Available Models
```bash
curl http://localhost:8000/models
```

#### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@audio.wav" \
  "?model_type=ensemble&ensemble_type=adaptive"
```

Response:
```json
{
  "filename": "audio.wav",
  "prediction": "REAL",
  "confidence": 0.95,
  "model_type": "ensemble",
  "ensemble_type": "adaptive",
  "status": "success"
}
```

---

## 📁 Project Structure

```
audio-deepfake-detection/
├── 🎵 app.py                     # Streamlit web application
├── 🔌 main.py                    # FastAPI server
├── 🔧 init_models.py             # Model initialization script
│
├── 📚 src/
│   ├── model.py                  # CNN architectures
│   ├── ensemble.py               # Ensemble models
│   ├── train.py                  # Training pipeline
│   ├── eval.py                   # Evaluation metrics
│   ├── augmentation.py           # Data augmentation
│   ├── optimization.py           # Model optimization
│   └── __pycache__/
│
├── 💾 model/
│   ├── enhanced_model.pth        # Enhanced model (15.09 MB)
│   ├── lightweight_model.pth     # Lightweight model (0.40 MB)
│   └── ensemble_*.pth            # Ensemble models
│
├── ⚙️ .streamlit/
│   └── config.toml               # Streamlit settings
│
├── 📋 config.yaml                # System configuration
├── 📖 README.md                  # This file
├── 📖 SETUP.md                   # Detailed setup guide
├── 📦 requirements.txt           # Python dependencies
├── 🔧 setup.py                   # Setup script
├── 🛠️ utils.py                   # Utility functions
└── .venv/                        # Virtual environment
```

---

## 🔧 Requirements

- **Python**: 3.8+
- **CUDA**: Optional (for GPU acceleration)
- **RAM**: 8GB minimum
- **Storage**: 500MB+

See `requirements.txt` for full list of dependencies.

---

## 📚 Architecture Details

### Enhanced CNN
- **Input**: 3 feature types (128×128 spectrograms)
- **Processing**: Residual blocks with self-attention
- **Output**: Binary classification (Real/Fake)

### Ensemble Methods
1. **Standard**: Weighted averaging of model outputs
2. **MultiScale**: Multi-resolution temporal processing
3. **Adaptive**: Dynamic weighting based on input characteristics

See [src/model.py](src/model.py) and [src/ensemble.py](src/ensemble.py) for implementation details.

---

## 🎓 Training Custom Models

To train with your own data:

```bash
python src/train.py
```

Configuration in `config.yaml`:
- Model hyperparameters
- Data augmentation settings
- Training schedule
- Audio processing parameters

See [SETUP.md](SETUP.md) for detailed training guide.

---

## 📊 Performance Metrics

The system evaluates models using:
- **Accuracy**: Percentage of correct predictions
- **Precision/Recall**: Per-class performance
- **ROC-AUC**: Discrimination ability
- **EER**: Equal Error Rate (anti-spoofing metric)

Evaluation tools in [src/eval.py](src/eval.py)

---

## 🐛 Troubleshooting

### Models not found
```bash
python init_models.py
```

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

See [SETUP.md](SETUP.md) for more troubleshooting tips.

---

## 📝 Dataset

- **ASVspoof 2019 LA Database**
- 51,000+ speech samples
- Multiple synthetic speech types
- Voice conversion techniques
- Real and spoofed utterances

Learn more: https://www.asvspoof.org/

---

## 🔍 Audio Features Explained

### Mel-Spectrogram
Frequency domain representation using mel-scale (human hearing perception). Better for detecting subtle audio artifacts.

### MFCC (Mel-Frequency Cepstral Coefficients)
Captures phonetic information. Includes first and second derivatives for temporal dynamics.

### Phase Features
Instantaneous frequency analysis. Useful for detecting unnatural phase coherence in synthetic speech.

---

## 💡 How It Works

1. **Audio Input**: User uploads WAV or MP3 file
2. **Feature Extraction**: Convert to 3 audio feature types
3. **Model Processing**: Run through selected model/ensemble
4. **Classification**: Output probability for Real/Fake
5. **Visualization**: Display waveform, spectrogram, and results

---

## 🚀 Deployment

### Streamlit Cloud
```bash
streamlit run app.py
```

### Docker (Coming Soon)
```bash
docker build -t audio-deepfake .
docker run -p 8501:8501 audio-deepfake
```

### Kubernetes
Configuration files available for production deployment.

---

## 📄 License

[Specify your license here]

---

## 👥 Authors

[Your name/organization]

---

## 🙏 Acknowledgments

- ASVspoof Challenge organizers
- PyTorch community
- Librosa audio processing library

---

## 📞 Support

For issues, questions, or contributions:
1. Check [SETUP.md](SETUP.md) for common issues
2. Review [src/](src/) for technical details
3. Open an issue on GitHub

---

## ⭐ Quick Links

- 📖 [Setup Guide](SETUP.md)
- 🎯 [Configuration](config.yaml)
- 📚 [Model Details](src/model.py)
- 🔬 [Training Guide](src/train.py)
- 📊 [Evaluation](src/eval.py)

---

**Ready to detect deepfakes? Start with:**
```bash
streamlit run app.py
```

Happy 🎵 analyzing!
