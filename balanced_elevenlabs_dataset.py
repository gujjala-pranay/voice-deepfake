"""
Balanced Dataset Loader for ElevenLabs Deepfake Dataset
Includes WeightedRandomSampler and Class Weights for imbalanced data.
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
from collections import Counter

class ElevenLabsDataset(Dataset): 
    def __init__(self, data_root, max_len=32000, sample_rate=16000,
                 augment=False, file_extensions=None, cache_dir=None):

        self.data_root = Path(data_root)
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.augment = augment

        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.data_root / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        if file_extensions is None:
            file_extensions = ['.wav', '.flac', '.mp3', '.ogg']
        self.file_extensions = [ext.lower() for ext in file_extensions]

        self.file_list = []
        self.labels = []
        self._build_file_list()

        print("\nElevenLabs Dataset Loaded:")
        print(f"   Total samples: {len(self.file_list)}")
        self.class_counts = Counter(self.labels)
        print(f"   Real (1): {self.class_counts[1]}")
        print(f"   Fake (0): {self.class_counts[0]}")

        # Augmentation
        if self.augment:
            try:
                import audiomentations
                self.audio_augment = audiomentations.Compose([
                    audiomentations.AddGaussianNoise(p=0.3),
                    audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                    audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                ])
            except ImportError:
                print("⚠️ audiomentations not installed — disabling augmentation")
                self.augment = False

    def _build_file_list(self):
        folder_labels = {
            'Original': 1,
            'ElevenLabs': 0,
            'Tacotron': 0,
            'Text To Speech': 0,
            'Voice Conversion': 0,
        }

        for folder_name, label in folder_labels.items():
            folder_path = self.data_root / folder_name
            if not folder_path.exists():
                continue

            for ext in self.file_extensions:
                for file_path in folder_path.glob(f'*{ext}'):
                    self.file_list.append(str(file_path))
                    self.labels.append(label)

        combined = list(zip(self.file_list, self.labels))
        np.random.seed(42)
        np.random.shuffle(combined)

        if combined:
            self.file_list, self.labels = zip(*combined)
        else:
            self.file_list, self.labels = [], []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        cache_path = self.cache_dir / (Path(file_path).stem + ".pt")

        if cache_path.exists():
            try:
                sample = torch.load(cache_path)
                # Ensure label is correct in case of cache reuse across different tasks
                sample['label'] = torch.tensor(label, dtype=torch.long)
                return sample
            except:
                pass # If cache is corrupted, re-extract

        try:
            audio, sr = sf.read(file_path)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            if len(audio) < self.max_len:
                audio = np.pad(audio, (0, self.max_len - len(audio)), 'constant')
            else:
                audio = audio[:self.max_len]
        except Exception:
            audio = np.zeros(self.max_len)

        # Apply augmentation
        if self.augment:
            audio = self.audio_augment(samples=audio, sample_rate=self.sample_rate)

        features = self._extract_features(audio)

        sample = {
            'spectral': torch.FloatTensor(features['spectral']),
            'mfcc': torch.FloatTensor(features['mfcc']),
            'phase': torch.FloatTensor(features['phase']),
            'label': torch.tensor(label, dtype=torch.long),
            'filename': os.path.basename(file_path)
        }

        torch.save(sample, cache_path)
        return sample

    def _extract_features(self, audio):
        return {
            'spectral': self._extract_spectral(audio),
            'mfcc': self._extract_mfcc(audio),
            'phase': self._extract_phase(audio)
        }

    def _extract_spectral(self, audio):
        features = []
        target_frames = 128
        for n_fft in [512, 1024, 2048]:
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=128
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            if mel_db.shape[1] < target_frames:
                mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), 'constant')
            else:
                mel_db = mel_db[:, :target_frames]
            features.append(mel_db)
        return np.stack(features, axis=0)

    def _extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        def fix(feat):
            if feat.shape[1] < 128:
                return np.pad(feat, ((0, 0), (0, 128 - feat.shape[1])), 'constant')
            return feat[:, :128]
        return np.stack([fix(mfcc), fix(mfcc_delta), fix(mfcc_delta2)], axis=0)

    def _extract_phase(self, audio):
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        phase = np.angle(stft)
        unwrapped = np.unwrap(phase)
        inst_freq = np.diff(unwrapped, axis=1)
        phase_features = np.abs(inst_freq)
        if phase_features.shape[0] < 128:
            phase_features = np.pad(phase_features, ((0, 128 - phase_features.shape[0]), (0, 0)), 'constant')
        else:
            phase_features = phase_features[:128, :]
        if phase_features.shape[1] < 128:
            phase_features = np.pad(phase_features, ((0, 0), (0, 128 - phase_features.shape[1])), 'constant')
        else:
            phase_features = phase_features[:, :128]
        return phase_features.reshape(1, 128, 128)

def get_balanced_loaders(data_root, batch_size=16, val_split=0.2, augment=True, num_workers=0, cache_dir=None):
    full_dataset = ElevenLabsDataset(data_root=data_root, augment=augment, cache_dir=cache_dir)
    
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False

    # Calculate weights for balancing the training set
    train_indices = train_dataset.indices
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = Counter(train_labels)
    
    # Weight for each class: 1 / count
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    # Weight for each sample in the training set
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    # Also return class weights for loss function if needed
    total_train = sum(class_counts.values())
    norm_weights = torch.tensor([total_train / (len(class_counts) * class_counts[0]), 
                                 total_train / (len(class_counts) * class_counts[1])])

    return train_loader, val_loader, norm_weights
