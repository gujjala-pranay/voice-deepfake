import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SelfAttention(nn.Module):
    """Self-attention mechanism for feature refinement"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Compute attention maps
        proj_query = self.query(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention
        proj_value = self.value(x).view(batch_size, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class DeepfakeDetectorCNN(nn.Module):
    """
    Enhanced 2D CNN architecture with Residual Connections and Attention.
    Input: Multi-Feature Representations (Spectral, MFCC, Phase)
    Output: Binary Classification (Real/Fake)
    """
    def __init__(self, num_classes=2):
        super(DeepfakeDetectorCNN, self).__init__()
        
        # Feature extraction branches
        self.spectral_branch = self._make_branch(3, 64)  # Multi-scale spectral features
        self.mfcc_branch = self._make_branch(3, 64)     # MFCC features
        self.phase_branch = self._make_branch(1, 64)     # Phase features
        
        # Attention mechanisms for each branch
        self.spectral_attention = SelfAttention(256)
        self.mfcc_attention = SelfAttention(256)
        self.phase_attention = SelfAttention(256)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(768, 256, 1)  # 256*3 = 768
        self.fusion_bn = nn.BatchNorm2d(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def _make_branch(self, in_channels, out_channels):
        """Create a CNN branch with residual blocks"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels, out_channels * 2, stride=2),  # Downsample
            ResidualBlock(out_channels * 2, out_channels * 4, stride=2),  # Downsample
        )
    
    def forward(self, spectral, mfcc, phase):
        # Process each feature branch
        spec_out = self.spectral_branch(spectral)      # (B, 256, H, W)
        mfcc_out = self.mfcc_branch(mfcc)             # (B, 256, H, W)
        phase_out = self.phase_branch(phase)          # (B, 256, H, W)
        
        # Apply attention
        spec_out = self.spectral_attention(spec_out)
        mfcc_out = self.mfcc_attention(mfcc_out)
        phase_out = self.phase_attention(phase_out)
        
        # Ensure same spatial dimensions for fusion
        if spec_out.shape[2:] != mfcc_out.shape[2:] or spec_out.shape[2:] != phase_out.shape[2:]:
            # Interpolate to match dimensions
            target_size = (8, 8)  # Common size
            spec_out = F.interpolate(spec_out, size=target_size, mode='bilinear', align_corners=False)
            mfcc_out = F.interpolate(mfcc_out, size=target_size, mode='bilinear', align_corners=False)
            phase_out = F.interpolate(phase_out, size=target_size, mode='bilinear', align_corners=False)
        
        # Feature fusion
        fused_features = torch.cat([spec_out, mfcc_out, phase_out], dim=1)  # (B, 768, H, W)
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.fusion_bn(fused_features)
        fused_features = F.relu(fused_features)
        
        # Global pooling and classification
        pooled = self.global_pool(fused_features)  # (B, 256, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)    # (B, 256)
        output = self.classifier(pooled)
        
        return output

class LightweightDetector(nn.Module):
    """Lightweight version for faster inference"""
    def __init__(self, num_classes=2):
        super(LightweightDetector, self).__init__()
        
        # Simplified architecture for speed
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # Input: multi-scale features
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # For lightweight version, use only spectral features
        features = self.features(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)

def get_model(model_type='enhanced', num_classes=2):
    """Factory function to get different model variants"""
    if model_type == 'enhanced':
        return DeepfakeDetectorCNN(num_classes)
    elif model_type == 'lightweight':
        return LightweightDetector(num_classes)
    else:
        return DeepfakeDetectorCNN(num_classes)

def compile_model(model):
    """
    Torch compilation for speedup (PyTorch 2.0+)
    Disabled on Windows without C++ compiler
    """
    try:
        # Check if we're on Windows and compiler is available
        import platform
        if platform.system() == 'Windows':
            # Try to compile but catch errors
            compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            return compiled_model
        else:
            compiled_model = torch.compile(model, mode='reduce-overhead')
            return compiled_model
    except Exception as e:
        print(f"Model compilation skipped: {e}")
        return model

def quantize_model(model):
    """Quantize model for faster inference"""
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model
