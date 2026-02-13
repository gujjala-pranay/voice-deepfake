import torch
import torch.nn as nn
import numpy as np
import librosa
import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import random
import warnings
warnings.filterwarnings('ignore')

class AudioAugmentation:
    """
    Comprehensive audio augmentation for deepfake detection
    """
    def __init__(self, sample_rate=16000, augment_prob=0.5):
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob
        
        # Basic audio augmentations
        # Shift parameters have changed in newer audiomentations versions
        # Using alternative: TimeStretch for time shifting
        self.basic_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
        ])
        
        # Advanced augmentations - removed Shift due to API changes
        self.advanced_augment = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.4),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.4),
            PitchShift(min_semitones=-6, max_semitones=6, p=0.4),
        ])
        
    def __call__(self, audio):
        """Apply random augmentation to audio"""
        if random.random() < self.augment_prob:
            # Choose between basic and advanced augmentation
            if random.random() < 0.7:
                return self.basic_augment(audio, sample_rate=self.sample_rate)
            else:
                return self.advanced_augment(audio, sample_rate=self.sample_rate)
        return audio

class SpectralAugmentation:
    """
    Spectral augmentation techniques (SpecAugment variants)
    """
    def __init__(self, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        
    def __call__(self, spectrogram):
        """
        Apply SpecAugment to spectrogram
        Args:
            spectrogram: (C, H, W) or (H, W) numpy array
        """
        aug_spec = spectrogram.copy()
        
        # Handle both 2D and 3D spectrograms
        if len(aug_spec.shape) == 3:
            # Apply to each channel
            for c in range(aug_spec.shape[0]):
                aug_spec[c] = self._apply_spec_augment(aug_spec[c])
        else:
            aug_spec = self._apply_spec_augment(aug_spec)
            
        return aug_spec
    
    def _apply_spec_augment(self, spec):
        """Apply SpecAugment to single channel spectrogram"""
        H, W = spec.shape
        
        # Time masking
        for _ in range(self.num_time_masks):
            if W > self.time_mask_param:
                mask_width = random.randint(0, self.time_mask_param)
                mask_start = random.randint(0, W - mask_width)
                spec[:, mask_start:mask_start + mask_width] = 0
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            if H > self.freq_mask_param:
                mask_height = random.randint(0, self.freq_mask_param)
                mask_start = random.randint(0, H - mask_height)
                spec[mask_start:mask_start + mask_height, :] = 0
        
        return spec

class MixupAugmentation:
    """
    Mixup augmentation for better generalization
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_data, batch_labels):
        """
        Apply mixup to a batch of data
        Args:
            batch_data: (B, C, H, W) tensor
            batch_labels: (B,) tensor
        Returns:
            mixed_data: (B, C, H, W) tensor
            labels_a: (B,) tensor
            labels_b: (B,) tensor
            lam: (B,) tensor
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size)
        
        mixed_data = lam * batch_data + (1 - lam) * batch_data[index]
        
        labels_a = batch_labels
        labels_b = batch_labels[index]
        
        return mixed_data, labels_a, labels_b, lam

class CutMixAugmentation:
    """
    CutMix augmentation for better robustness
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch_data, batch_labels):
        """
        Apply CutMix to a batch of data
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size)
        
        _, _, H, W = batch_data.shape
        
        # Generate random bounding box
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Random position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        batch_data[:, :, bby1:bby2, bbx1:bbx2] = batch_data[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_a = batch_labels
        labels_b = batch_labels[index]
        
        return batch_data, labels_a, labels_b, lam

class AdaptiveAugmentation:
    """
    Adaptive augmentation based on sample difficulty
    """
    def __init__(self, base_augmentation, difficulty_threshold=0.7):
        self.base_augmentation = base_augmentation
        self.difficulty_threshold = difficulty_threshold
        
    def __call__(self, audio, confidence_score=None):
        """
        Apply stronger augmentation to difficult samples
        """
        if confidence_score is not None and confidence_score < self.difficulty_threshold:
            # Apply stronger augmentation for difficult samples
            return self._apply_strong_augmentation(audio)
        else:
            return self.base_augmentation(audio)
    
    def _apply_strong_augmentation(self, audio):
        """Apply stronger augmentation - without Shift due to API changes"""
        strong_augment = Compose([
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=0.6),
            TimeStretch(min_rate=0.6, max_rate=1.4, p=0.6),
            PitchShift(min_semitones=-8, max_semitones=8, p=0.6),
        ])
        return strong_augment(audio, sample_rate=16000)

class FeatureAugmentation:
    """
    Augmentation at feature level
    """
    def __init__(self, noise_std=0.01, dropout_prob=0.1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        
    def __call__(self, features):
        """
        Apply feature-level augmentation
        Args:
            features: (C, H, W) numpy array or tensor
        """
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add Gaussian noise
        noise = torch.randn_like(features) * self.noise_std
        noisy_features = features + noise
        
        # Feature dropout
        mask = torch.rand_like(features) > self.dropout_prob
        augmented_features = noisy_features * mask
        
        return augmented_features

class CompositeAugmentation:
    """
    Composite augmentation combining multiple techniques
    """
    def __init__(self, audio_aug_prob=0.5, spectral_aug_prob=0.3, mixup_prob=0.2):
        self.audio_augmentation = AudioAugmentation(augment_prob=audio_aug_prob)
        self.spectral_augmentation = SpectralAugmentation()
        self.mixup_augmentation = MixupAugmentation()
        self.cutmix_augmentation = CutMixAugmentation()
        
        self.audio_aug_prob = audio_aug_prob
        self.spectral_aug_prob = spectral_aug_prob
        self.mixup_prob = mixup_prob
        
    def augment_sample(self, audio, spectral_features=None):
        """
        Augment a single sample
        """
        # Audio-level augmentation
        if random.random() < self.audio_aug_prob:
            audio = self.audio_augmentation(audio)
        
        # Spectral-level augmentation
        if spectral_features is not None and random.random() < self.spectral_aug_prob:
            spectral_features = self.spectral_augmentation(spectral_features)
        
        return audio, spectral_features
    
    def augment_batch(self, batch_data, batch_labels):
        """
        Augment a batch of data
        """
        # Randomly choose between mixup and cutmix
        if random.random() < self.mixup_prob:
            return self.mixup_augmentation(batch_data, batch_labels)
        else:
            return self.cutmix_augmentation(batch_data, batch_labels)

# Utility functions
def create_augmentation_pipeline(augmentation_type='composite', **kwargs):
    """
    Factory function to create augmentation pipeline
    """
    if augmentation_type == 'basic':
        return AudioAugmentation(**kwargs)
    elif augmentation_type == 'spectral':
        return SpectralAugmentation(**kwargs)
    elif augmentation_type == 'mixup':
        return MixupAugmentation(**kwargs)
    elif augmentation_type == 'cutmix':
        return CutMixAugmentation(**kwargs)
    elif augmentation_type == 'adaptive':
        return AdaptiveAugmentation(AudioAugmentation(), **kwargs)
    elif augmentation_type == 'composite':
        return CompositeAugmentation(**kwargs)
    else:
        return AudioAugmentation(**kwargs)

def apply_augmentation_to_features(features_dict, augmentation_pipeline):
    """
    Apply augmentation to extracted features
    """
    augmented_features = {}
    
    for key, features in features_dict.items():
        if key in ['spectral', 'mfcc', 'phase']:
            # Apply spectral augmentation
            if hasattr(augmentation_pipeline, 'spectral_augmentation'):
                augmented_features[key] = augmentation_pipeline.spectral_augmentation(features)
            else:
                augmented_features[key] = features
        else:
            augmented_features[key] = features
    
    return augmented_features

# Training utilities
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss function for mixup augmentation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss function for cutmix augmentation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
