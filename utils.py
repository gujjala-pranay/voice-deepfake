import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import librosa.display

class AudioDataset(Dataset):
    """
    Custom Dataset for ASVspoof 2019.
    Loads audio files and converts them to Mel-Spectrograms.
    """
    def __init__(self, protocol_file, data_dir, transform=None, max_len=64000, limit=10000):
        self.data_dir = data_dir
        self.transform = transform
        self.max_len = max_len
        self.file_list = []
        self.labels = []
        
        if os.path.exists(protocol_file):
            with open(protocol_file, 'r') as f:
                lines = f.readlines()
                # Use a medium dataset size
                if limit and len(lines) > limit:
                    lines = lines[:limit]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        file_name = parts[1]
                        # Label: 1 for bonafide (REAL), 0 for spoof (FAKE)
                        label = 1 if parts[4] == 'bonafide' else 0
                        self.file_list.append(file_name)
                        self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx] + '.wav')
        try:
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            # Return a zero array if file loading fails
            audio = np.zeros(self.max_len)
            sr = 16000
        
        # Padding or Truncating to ensure fixed length
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), 'constant')
        else:
            audio = audio[:self.max_len]
            
        # Feature Extraction: Mel-Spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Add channel dimension (C, H, W)
        spectrogram_db = np.expand_dims(spectrogram_db, axis=0)
        
        if self.transform:
            spectrogram_db = self.transform(spectrogram_db)
            
        return torch.FloatTensor(spectrogram_db), torch.tensor(self.labels[idx], dtype=torch.long)

def plot_spectrogram(audio_path, title="Mel-Spectrogram"):
    """
    Generates a Mel-Spectrogram plot for a given audio file.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    return fig

def plot_waveform(audio_path, title="Waveform"):
    """
    Generates a waveform plot for a given audio file.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set(title=title)
    return fig
