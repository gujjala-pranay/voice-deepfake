import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import librosa.display
import hashlib
from functools import lru_cache
import audiomentations
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class AudioDataset(Dataset):
    """
    Enhanced Dataset for ASVspoof 2019 with multi-feature extraction.
    Loads audio files and converts them to multiple feature representations.
    """
    def __init__(self, data_source, data_dir=None, transform=None,max_len=64000, limit=10000, augment=False):

        self.transform = transform
        self.max_len = max_len
        self.file_list = []
        self.labels = []
        self.augment = augment

        self.audio_augment = audiomentations.Compose([
            audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            audiomentations.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
        ])

        # CASE 1: Protocol-based dataset
        if isinstance(data_source, str) and data_dir is not None:
            self.data_dir = data_dir

            if os.path.exists(data_source):
                with open(data_source, 'r') as f:
                    lines = f.readlines()
                    if limit and len(lines) > limit:
                        lines = lines[:limit]

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            file_name = parts[1]
                            label = 1 if parts[4] == 'bonafide' else 0
                            self.file_list.append(file_name)
                            self.labels.append(label)

        # CASE 2: Direct file list [(filepath, label)]
        elif isinstance(data_source, list):
            self.data_dir = None
            for filepath, label in data_source:
                self.file_list.append(filepath)
                self.labels.append(label)

        else:
            raise ValueError("Invalid data source for AudioDataset")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.data_dir is not None:
            base_path = os.path.join(self.data_dir, self.file_list[idx])
            file_path = base_path + '.flac'
            if not os.path.exists(file_path):
                file_path = base_path + '.wav'
        else:
            file_path = self.file_list[idx]


        
        if not os.path.exists(file_path):
            file_path = base_path + '.wav'
        
        try:
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            # Return a zero array if file loading fails
            audio = np.zeros(self.max_len)
            sr = 16000
        
        # Apply audio augmentation if enabled
        if self.augment and np.random.random() > 0.5:
            audio = self.audio_augment(samples=audio, sample_rate=sr)
        
        # Padding or Truncating to ensure fixed length
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), 'constant')
        else:
            audio = audio[:self.max_len]
            
        # Extract multiple features
        features = self.extract_all_features(audio, sr)
        
        # Apply spec augmentation if transform is provided
        if self.transform:
            features['spectral'] = self.transform(features['spectral'])
            features['mfcc'] = self.transform(features['mfcc'])
            
        return {
            'spectral': torch.FloatTensor(features['spectral']),
            'mfcc': torch.FloatTensor(features['mfcc']),
            'phase': torch.FloatTensor(features['phase']),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def extract_all_features(self, audio, sr=16000):
        """Extract all feature types for enhanced detection"""
        return {
            'spectral': self.extract_multi_scale_features(audio, sr),
            'mfcc': self.extract_mfcc_delta(audio, sr),
            'phase': self.extract_phase_features(audio, sr)
        }
    
    def extract_multi_scale_features(self, audio, sr=16000):
        """Extract multi-scale mel-spectrograms"""
        features = []
        target_time_frames = 128
        
        # Different window sizes for temporal resolution
        for n_fft in [512, 1024, 2048]:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=n_fft, hop_length=n_fft//4, n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize time dimension to match target
            if mel_spec_db.shape[1] < target_time_frames:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_time_frames - mel_spec_db.shape[1])), 'constant')
            else:
                mel_spec_db = mel_spec_db[:, :target_time_frames]
            
            features.append(mel_spec_db)
        
        return np.stack(features, axis=0)  # (3, 128, 128)
    
    def extract_mfcc_delta(self, audio, sr=16000):
        """Extract MFCC features with delta and delta-delta"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Pad to match spectral features size
        target_length = 128
        if mfcc.shape[1] < target_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), 'constant')
            mfcc_delta = np.pad(mfcc_delta, ((0, 0), (0, target_length - mfcc_delta.shape[1])), 'constant')
            mfcc_delta2 = np.pad(mfcc_delta2, ((0, 0), (0, target_length - mfcc_delta2.shape[1])), 'constant')
        else:
            mfcc = mfcc[:, :target_length]
            mfcc_delta = mfcc_delta[:, :target_length]
            mfcc_delta2 = mfcc_delta2[:, :target_length]
        
        return np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=0)  # (3, 20, T)
    
    def extract_phase_features(self, audio, sr=16000):
        """Extract phase-based features"""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        phase = np.angle(stft)
        unwrapped_phase = np.unwrap(phase)
        
        # Phase derivative (instantaneous frequency)
        inst_freq = np.diff(unwrapped_phase, axis=1)
        phase_features = np.abs(inst_freq)
        
        # Resize to match other features
        target_shape = (1, 128, 128)
        if phase_features.shape[0] < target_shape[1]:
            phase_features = np.pad(phase_features, ((0, target_shape[1] - phase_features.shape[0]), (0, 0)), 'constant')
        else:
            phase_features = phase_features[:target_shape[1], :]
            
        if phase_features.shape[1] < target_shape[2]:
            phase_features = np.pad(phase_features, ((0, 0), (0, target_shape[2] - phase_features.shape[1])), 'constant')
        else:
            phase_features = phase_features[:, :target_shape[2]]
        
        return phase_features.reshape(target_shape)
    
    @staticmethod
    def spec_augment(spectrogram, time_mask=20, freq_mask=10):
        """Apply SpecAugment to spectrogram"""
        aug_spec = spectrogram.copy()
        
        # Time masking
        if aug_spec.shape[2] > time_mask:
            time_start = np.random.randint(0, aug_spec.shape[2] - time_mask)
            aug_spec[:, :, time_start:time_start + time_mask] = 0
        
        # Frequency masking
        if aug_spec.shape[1] > freq_mask:
            freq_start = np.random.randint(0, aug_spec.shape[1] - freq_mask)
            aug_spec[:, freq_start:freq_start + freq_mask, :] = 0
        
        return aug_spec
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_feature_extraction(audio_path):
        """Cache feature extraction based on file hash"""
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_key = f"{audio_path}_{file_hash}"
        # Implement caching logic here
        return cache_key

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
