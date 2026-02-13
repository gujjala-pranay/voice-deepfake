import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import DeepfakeDetectorCNN, LightweightDetector, get_model
import numpy as np

class EnsembleDetector(nn.Module):
    """
    Fixed Ensemble with proper calibration and bias correction
    """
    def __init__(self, model_types=['enhanced', 'lightweight'], weights=None):
        super(EnsembleDetector, self).__init__()
        
        self.models = nn.ModuleList()
        self.model_types = model_types
        
        # Initialize different model variants
        for model_type in model_types:
            model = get_model(model_type)
            self.models.append(model)
        
        # Learnable ensemble weights (start with equal weights)
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(model_types)) / len(model_types))
        else:
            self.weights = nn.Parameter(torch.tensor(weights))
        
        # Bias correction parameter
        self.bias_correction = nn.Parameter(torch.zeros(1))
        
        # Simple fusion layer instead of complex meta-classifier
        self.fusion = nn.Sequential(
            nn.Linear(len(model_types) * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        
        # Initialize with better starting values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to avoid bias"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Start with equal weights
        with torch.no_grad():
            self.weights.fill_(1.0 / len(self.models))
            self.bias_correction.fill_(0.0)
    
    def forward(self, spectral, mfcc, phase):
        model_outputs = []
        model_probs = []
        
        for i, model in enumerate(self.models):
            if isinstance(model, LightweightDetector):
                # Lightweight model only uses spectral features
                output = model(spectral)
            else:
                # Enhanced model uses all features
                output = model(spectral, mfcc, phase)
            
            probs = F.softmax(output, dim=1)
            model_outputs.append(output)
            model_probs.append(probs)
        
        # Apply learnable weights with softmax normalization
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Weighted averaging of probabilities
        weighted_probs = torch.zeros_like(model_probs[0])
        for i, probs in enumerate(model_probs):
            weighted_probs += normalized_weights[i] * probs
        
        # Concatenate all probabilities for fusion
        concat_probs = torch.cat(model_probs, dim=1)
        
        # Fusion layer
        fused_output = self.fusion(concat_probs)
        
        # Blend weighted average and fusion (with bias correction)
        final_output = 0.6 * weighted_probs + 0.4 * F.softmax(fused_output, dim=1)
        
        # Apply bias correction
        final_output[:, 0] += self.bias_correction  # FAKE class
        final_output[:, 1] -= self.bias_correction  # REAL class
        
        # Renormalize
        final_output = F.softmax(final_output, dim=1)
        
        return final_output

class MultiScaleEnsemble(nn.Module):
    """
    Simplified ensemble with multiple model outputs
    """
    def __init__(self):
        super(MultiScaleEnsemble, self).__init__()
        
        # Three enhanced models - no complex scale processing
        self.model1 = get_model('enhanced')
        self.model2 = get_model('enhanced')
        self.model3 = get_model('enhanced')
        
        # Simple weighted fusion
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
        # Optional refinement layer
        self.refiner = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, spectral, mfcc, phase):
        # Get outputs from all three models
        output1 = self.model1(spectral, mfcc, phase)
        output2 = self.model2(spectral, mfcc, phase)
        output3 = self.model3(spectral, mfcc, phase)
        
        # Get probabilities
        prob1 = F.softmax(output1, dim=1)
        prob2 = F.softmax(output2, dim=1)
        prob3 = F.softmax(output3, dim=1)
        
        # Stack and apply learnable weights
        all_probs = torch.stack([prob1, prob2, prob3], dim=0)  # (3, B, 2)
        weighted_probs = (all_probs * self.weights.view(3, 1, 1)).sum(dim=0)  # (B, 2)
        
        # Average the logits for final output
        combined_logits = (output1 + output2 + output3) / 3
        
        # Refine with weighted probabilities
        refined_output = self.refiner(weighted_probs)
        
        # Blend logits and refined output
        final_output = 0.6 * combined_logits + 0.4 * refined_output
        
        return final_output

class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that weights models based on input characteristics
    """
    def __init__(self):
        super(AdaptiveEnsemble, self).__init__()
        
        # Base models
        self.models = nn.ModuleList([
            get_model('enhanced'),
            get_model('lightweight')
        ])
        
        # Input characteristic analyzer
        self.input_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, len(self.models))  # Output weights for each model
        )
        
        # Final classifier
        self.final_classifier = nn.Linear(2, 2)  # Combine weighted outputs
        
    def forward(self, spectral, mfcc, phase):
        # Analyze input to determine model weights
        input_weights = self.input_analyzer(spectral)
        input_weights = F.softmax(input_weights, dim=1)
        
        model_outputs = []
        for i, model in enumerate(self.models):
            if isinstance(model, LightweightDetector):
                output = model(spectral)
            else:
                output = model(spectral, mfcc, phase)
            
            # Weight by input characteristics
            weighted_output = input_weights[:, i:i+1] * output
            model_outputs.append(weighted_output)
        
        # Combine weighted outputs
        combined_output = torch.stack(model_outputs, dim=0).sum(dim=0)
        final_output = self.final_classifier(combined_output)
        
        return final_output

def create_ensemble(ensemble_type='standard'):
    """Factory function to create different ensemble types"""
    if ensemble_type == 'standard':
        return EnsembleDetector(['enhanced', 'lightweight'])
    elif ensemble_type == 'multiscale':
        return MultiScaleEnsemble()
    elif ensemble_type == 'adaptive':
        return AdaptiveEnsemble()
    else:
        return EnsembleDetector()

# Utility functions for ensemble training and evaluation
def train_ensemble(ensemble_model, train_loader, val_loader, epochs=10, lr=0.001, device='cuda'):
    """Training function specifically for ensemble models"""
    ensemble_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        ensemble_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                spectral = batch['spectral'].to(device)
                mfcc = batch['mfcc'].to(device)
                phase = batch['phase'].to(device)
                labels = batch['label'].to(device)
            else:
                # Legacy format
                inputs, labels = batch
                spectral, mfcc, phase = inputs
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ensemble_model(spectral, mfcc, phase)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        val_acc = evaluate_ensemble(ensemble_model, val_loader, device)
        
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ensemble_model.state_dict(), '../model/ensemble_model.pth')
            print("Ensemble model saved!")

def evaluate_ensemble(ensemble_model, loader, device):
    """Evaluation function for ensemble models"""
    ensemble_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                spectral = batch['spectral'].to(device)
                mfcc = batch['mfcc'].to(device)
                phase = batch['phase'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs, labels = batch
                spectral, mfcc, phase = inputs
                spectral, mfcc, phase = spectral.to(device), mfcc.to(device), phase.to(device)
                labels = labels.to(device)
            
            outputs = ensemble_model(spectral, mfcc, phase)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total
