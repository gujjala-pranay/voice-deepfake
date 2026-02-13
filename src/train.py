import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from src.model import DeepfakeDetectorCNN, get_model
from src.ensemble import create_ensemble, train_ensemble, evaluate_ensemble
from src.augmentation import create_augmentation_pipeline, mixup_criterion, cutmix_criterion
from src.optimization import create_optimized_model, Profiler, MemoryOptimizer
import sys
import os
sys.path.append('..')
from utils import AudioDataset
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import warnings
warnings.filterwarnings('ignore')

class EnhancedTrainer:
    """Enhanced training class with all optimizations"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = Profiler()
        self.memory_optimizer = MemoryOptimizer()
        
        # Initialize model
        if config['use_ensemble']:
            self.model = create_ensemble(config['ensemble_type'])
        else:
            self.model = get_model(config['model_type'])
            
        # Optimize model
        if config['optimize_model']:
            self.model = create_optimized_model(self.model, config['optimization_level'])
        
        self.model.to(self.device)
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        
        # Optimizer with weight decay
        if config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay']
            )
        
        # Learning rate scheduler
        if config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config['epochs'], eta_min=1e-6
            )
        elif config['scheduler'] == 'one_cycle':
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=config['learning_rate'], 
                steps_per_epoch=config['steps_per_epoch'], epochs=config['epochs']
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.use_amp = config['use_amp'] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Augmentation
        self.augmentation = create_augmentation_pipeline(
            config['augmentation_type'], 
            **config.get('augmentation_params', {})
        )
        
        # Gradient accumulation
        self.accumulation_steps = config['gradient_accumulation_steps']
        
    def train_epoch(self, train_loader):
        """Train for one epoch with all optimizations"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, dict):
                spectral = batch['spectral'].to(self.device)
                mfcc = batch['mfcc'].to(self.device)
                phase = batch['phase'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                # Legacy format
                inputs, labels = batch
                if isinstance(inputs, (list, tuple)):
                    spectral, mfcc, phase = inputs
                    spectral, mfcc, phase = spectral.to(self.device), mfcc.to(self.device), phase.to(self.device)
                else:
                    spectral = inputs.to(self.device)
                    mfcc = phase = None
                labels = labels.to(self.device)
            
            # Apply batch-level augmentation
            if self.config.get('use_batch_augmentation', False) and np.random.random() < 0.3:
                if isinstance(batch, dict):
                    augmented_data, labels_a, labels_b, lam = self.augmentation.augment_batch(
                        batch['spectral'], batch['label']
                    )
                    spectral = augmented_data.to(self.device)
                    labels_a = labels_a.to(self.device)
                    labels_b = labels_b.to(self.device)
                    use_mixup = True
                else:
                    use_mixup = False
            else:
                use_mixup = False
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                if mfcc is not None and phase is not None:
                    outputs = self.model(spectral, mfcc, phase)
                else:
                    outputs = self.model(spectral)
                
                if use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                
                # Gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # Statistics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Memory management
            if batch_idx % 100 == 0:
                self.memory_optimizer.optimize_memory_usage()
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        return avg_loss, train_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    spectral = batch['spectral'].to(self.device)
                    mfcc = batch['mfcc'].to(self.device)
                    phase = batch['phase'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    inputs, labels = batch
                    if isinstance(inputs, (list, tuple)):
                        spectral, mfcc, phase = inputs
                        spectral, mfcc, phase = spectral.to(self.device), mfcc.to(self.device), phase.to(self.device)
                    else:
                        spectral = inputs.to(self.device)
                        mfcc = phase = None
                    labels = labels.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    if mfcc is not None and phase is not None:
                        outputs = self.model(spectral, mfcc, phase)
                    else:
                        outputs = self.model(spectral)
                    
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_loss = running_loss / len(val_loader)
        
        return avg_loss, val_acc
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                elif isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Memory usage
            memory_stats = self.memory_optimizer.get_memory_usage()
            print(f'Memory Usage: {memory_stats["used_gb"]:.2f}GB ({memory_stats["percent"]:.1f}%)')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                model_dir = 'model'
                os.makedirs(model_dir, exist_ok=True)
                
                if self.config['use_ensemble']:
                    save_path = os.path.join(model_dir, 'best_ensemble_model.pth')
                else:
                    save_path = os.path.join(model_dir, 'best_model.pth')
                
                torch.save(self.model.state_dict(), save_path)
                
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.get('patience', 10):
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Print performance report
        self.profiler.print_performance_report()
        
        return best_val_acc

def get_default_config():
    """Get default training configuration"""
    return {
        # Model configuration
        'model_type': 'enhanced',
        'use_ensemble': False,
        'ensemble_type': 'standard',
        'optimize_model': False,
        'optimization_level': 'medium',
        
        # Training hyperparameters
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'gradient_accumulation_steps': 4,
        
        # Optimizer and scheduler
        'optimizer': 'adamw',
        'scheduler': 'reduce_on_plateau',
        
        # Augmentation
        'use_amp': True,
        'augmentation_type': 'composite',
        'use_batch_augmentation': True,
        'augmentation_params': {
            'audio_aug_prob': 0.5,
            'spectral_aug_prob': 0.3,
            'mixup_prob': 0.2
        },
        
        # Early stopping
        'patience': 15,
        
        # Data loading
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        
        # Dataset
        'max_len': 64000,
        'limit': 10000,
        'augment': True
    }

from sklearn.model_selection import train_test_split
import os

def create_data_loaders(config):

    data_dir = r'E:\deepfake\audio-deepfake-detection\elevenlabs_dataset'

    real_files = []
    fake_files = []

    # REAL = Original
    real_path = os.path.join(data_dir, 'Original')
    for file in os.listdir(real_path):
        if file.endswith('.wav'):
            real_files.append((os.path.join(real_path, file), 0))

    # FAKE = All other folders
    fake_folders = ['ElevenLabs', 'Tacotron', 'Text To Speech', 'Voice Conversion']

    for folder in fake_folders:
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                fake_files.append((os.path.join(folder_path, file), 1))

    all_files = real_files + fake_files

    print("Total Real:", len(real_files))
    print("Total Fake:", len(fake_files))

    train_files, val_files = train_test_split(
        all_files,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in all_files]
    )

    train_dataset = AudioDataset(
        train_files,
        max_len=config['max_len'],
        augment=config['augment']
    )

    val_dataset = AudioDataset(
        val_files,
        max_len=config['max_len'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Get configuration
    config = get_default_config()
    
    # You can override config here
    # config['epochs'] = 100
    # config['use_ensemble'] = True
    
    print("=== Enhanced Audio Deepfake Detection Training ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Model Type: {config['model_type']}")
    print(f"Ensemble: {config['use_ensemble']}")
    print(f"Optimization: {config['optimization_level']}")
    print(f"Mixed Precision: {config['use_amp']}")
    print(f"Augmentation: {config['augmentation_type']}")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = EnhancedTrainer(config)
    
    # Update config with data loader info
    config['steps_per_epoch'] = len(train_loader)
    
    # Start training
    print("Starting training...")
    best_val_acc = trainer.train(train_loader, val_loader)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to ../model/best_model.pth")
