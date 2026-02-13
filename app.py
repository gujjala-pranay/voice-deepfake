import streamlit as st
import torch
import numpy as np
import librosa
import os
from src.model import DeepfakeDetectorCNN, get_model
from src.calibration import RobustPredictor, explain_prediction, save_pretrained_weights
from utils import plot_spectrogram, plot_waveform, AudioDataset
import torch.nn.functional as F

# Page Configuration
st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")

# Initialize pretrained weights if needed
if not os.path.exists('model/enhanced_pretrained.pth'):
    save_pretrained_weights()

# Load Model with calibration
def get_model_status(model_type):
    """Check if model has trained weights"""
    trained_path = f'model/{model_type}_trained.pth'
    pretrained_path = f'model/{model_type}_pretrained.pth'
    
    if os.path.exists(trained_path):
        return "trained", trained_path
    elif os.path.exists(pretrained_path):
        return "pretrained", pretrained_path
    else:
        return "uninitialized", None

@st.cache_resource
def load_model_with_calibration(model_type='enhanced'):
    """Load model with robust predictor and calibration"""
    
    # Handle ensemble models
    if model_type.startswith('ensemble_'):
        from src.ensemble import create_ensemble
        ensemble_type = model_type.split('_')[1]
        model = create_ensemble(ensemble_type)
    else:
        model = get_model(model_type)
    
    # Try to load trained weights first, then pretrained
    trained_path = f'model/{model_type}_trained.pth'
    pretrained_path = f'model/{model_type}_pretrained.pth'
    
    if os.path.exists(trained_path):
        model.load_state_dict(torch.load(trained_path, map_location=torch.device('cpu')))
        print(f"Loaded trained weights from {trained_path}")
    elif os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
    else:
        # Initialize with reasonable weights to avoid bias
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Use small random values
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                # Initialize classifier biases to zero to avoid bias
                if 'classifier' in name or 'fc' in name or 'fusion' in name:
                    torch.nn.init.constant_(param, 0.0)
    
    model.eval()
    
    # Wrap with robust predictor
    predictor = RobustPredictor(model, device='cpu')
    
    return predictor

def preprocess_audio(audio_path, max_len=32000):
    """Extract all features from audio file"""
    audio, sr = librosa.load(audio_path, sr=16000)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
    else:
        audio = audio[:max_len]
    
    # Use AudioDataset to extract all features
    dataset = AudioDataset.__new__(AudioDataset)  # Create instance without calling __init__
    features = dataset.extract_all_features(audio, sr)
    
    # Convert to tensors and add batch dimension
    spectral = torch.FloatTensor(features['spectral']).unsqueeze(0)
    mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0)
    phase = torch.FloatTensor(features['phase']).unsqueeze(0)
    
    return spectral, mfcc, phase

# UI Header
st.title("🎙️ Audio Deepfake Detection System")
st.markdown("""
This application uses a **2D Convolutional Neural Network (CNN)** to distinguish between **Real (Bona fide)** and **Fake (Deepfake)** audio.
Upload an audio file to see the analysis.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("Model Selection")

# Radio buttons for model selection
model_options = {
    "enhanced": "Enhanced (Full Features - Best Accuracy)",
    "lightweight": "Lightweight (Fast - Spectral Only)",
    "ensemble_standard": "Ensemble Standard (Multi-Model)",
    "ensemble_multiscale": "Ensemble MultiScale (Advanced)",
    "ensemble_adaptive": "Ensemble Adaptive (Smart Selection)"
}

model_type = st.sidebar.radio(
    "Select Detection Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    help="Choose the AI model for deepfake detection"
)

# Model training status
st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

# Get status for current model
status, path = get_model_status(model_type)

# Display status with color
if status == "trained":
    st.sidebar.success(f"✅ {model_type}: Trained")
    st.sidebar.caption(f"Using: {path}")
elif status == "pretrained":
    st.sidebar.warning(f"⚠️ {model_type}: Pretrained Only")
    st.sidebar.caption(f"Path: {path}")
    st.sidebar.info("💡 Run training for better accuracy")
else:
    st.sidebar.error(f"❌ {model_type}: Not Initialized")
    st.sidebar.info("⚠️ Model will use random weights")

# Show all models status
with st.sidebar.expander("View All Models"):
    for m in model_options.keys():
        s, p = get_model_status(m)
        icon = "✅" if s == "trained" else "⚠️" if s == "pretrained" else "❌"
        st.write(f"{icon} {m}")

# Training Section
st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Model Training")

# Check if any model needs training
needs_training = any(get_model_status(m)[0] != "trained" for m in model_options.keys())

if needs_training:
    st.sidebar.warning("Models need training!")
    st.sidebar.caption("Training improves accuracy from ~60% to ~90%+")
    
    # Quick training button
    if st.sidebar.button("⚡ Quick Train (5 epochs)", key="quick_train"):
        with st.spinner("Training in progress... This may take a few minutes."):
            import subprocess
            result = subprocess.run(
                [".venv/Scripts/python.exe", "train_simple.py"],
                capture_output=True,
                text=True,
                env={**os.environ, "TORCH_COMPILE_DISABLE": "1"}
            )
            if result.returncode == 0:
                st.sidebar.success("✅ Training complete!")
                st.rerun()
            else:
                st.sidebar.error("Training failed. Check console.")
    
    # Full training button
    if st.sidebar.button("🔥 Full Train (50 epochs)", key="full_train"):
        st.sidebar.info("Run in terminal: `python train_elevenlabs.py --epochs 50`")
        st.sidebar.code("python train_elevenlabs.py --epochs 50", language="bash")
else:
    st.sidebar.success("✅ All models trained!")

st.sidebar.markdown("---")
st.sidebar.header("About the Project")
st.sidebar.info(f"""
- **Current Model**: {model_type}
- **Features Used**: {'All 3 (Full)' if model_type != 'lightweight' else 'Spectral only'}
- **Dataset**: ElevenLabs Deepfake
- **Goal**: Detect synthetic speech and voice conversion.
""")

# File Upload
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Visualization")
        st.audio(uploaded_file, format='audio/wav')
        
        # Waveform
        fig_wave = plot_waveform(temp_path)
        st.pyplot(fig_wave)
        
        # Spectrogram
        fig_spec = plot_spectrogram(temp_path)
        st.pyplot(fig_spec)
        
    with col2:
        st.subheader("Prediction Result")
        
        predictor = load_model_with_calibration(model_type)
        
        with st.spinner('Analyzing audio...'):
            spectral, mfcc, phase = preprocess_audio(temp_path)
                
            with torch.no_grad():
                # Use robust prediction with calibration
                result = predictor.predict(spectral, mfcc, phase, return_confidence=True)
                
                prediction = result['prediction']
                confidence = result['confidence']
                probabilities = result['probabilities']
                is_valid = result['is_valid']
                warnings = result['warnings']
                
            # Display Result
            if prediction.item() == 1:
                st.success("✅ RESULT: REAL (Bona fide)")
                st.balloons()
            else:
                st.error("❌ RESULT: FAKE (Deepfake)")
                st.warning("⚠️ This audio appears to be artificially generated")
                
            # Confidence display
            st.metric("Confidence Score", f"{confidence*100:.1f}%")
            
            # Confidence improvement tips
            if confidence < 0.7:
                with st.expander("⚠️ How to Improve Confidence"):
                    st.write("""
                    **Low confidence detected. Tips to improve:**
                    
                    1. **Use longer audio clips** (4+ seconds ideal)
                    2. **Ensure clear audio quality** (reduce background noise)
                    3. **Use trained models** (run training on dataset)
                    4. **Try different models** (ensemble often gives higher confidence)
                    5. **Check audio format** (16kHz WAV recommended)
                    
                    **Current dataset status:**
                    - 2,561 audio files available
                    - 736 REAL (Original), 1,825 FAKE
                    - Run `python train_simple.py` to train models
                    """)
            
            # Show probabilities
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.progress(probabilities[0, 0].item(), text=f"FAKE: {probabilities[0, 0].item()*100:.1f}%")
            with col_prob2:
                st.progress(probabilities[0, 1].item(), text=f"REAL: {probabilities[0, 1].item()*100:.1f}%")
            
            # Show warnings if any
            if not is_valid and warnings:
                st.warning("⚠️ Prediction may not be reliable:")
                for warning in warnings:
                    st.write(f"- {warning}")
            
            # Technical details
            with st.expander("Technical Details"):
                st.write(f"**Logits**: {result['logits'].numpy()}")
                st.write(f"**Softmax**: {probabilities.numpy()}")
                st.write(f"**Model**: {model_type}")
                
                # Feature analysis
                st.write("**Feature Statistics:**")
                st.write(f"- Spectral mean: {spectral.mean().item():.4f}, std: {spectral.std().item():.4f}")
                st.write(f"- MFCC mean: {mfcc.mean().item():.4f}, std: {mfcc.std().item():.4f}")
                st.write(f"- Phase mean: {phase.mean().item():.4f}, std: {phase.std().item():.4f}")
            
            st.markdown("---")
            st.write("**Technical Explanation:**")
            st.write("""
            The model analyzes multiple audio features:
            - **Multi-scale Spectrograms**: Captures frequency patterns at different time resolutions
            - **MFCC with Deltas**: Mel-frequency cepstral coefficients with temporal derivatives
            - **Phase Features**: Captures instantaneous frequency and phase information
            
            Deepfake audio often has subtle artifacts in these features that are invisible to human hearing
            but detectable by the neural network's pattern recognition capabilities.
            """)
            
            # Training recommendation
            status, _ = get_model_status(model_type)
            if status != "trained":
                st.info("💡 **Tip**: This model is using pretrained/untrained weights. For best results, train on the ElevenLabs dataset using `python train_simple.py`")
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info("Please upload an audio file to begin analysis.")
