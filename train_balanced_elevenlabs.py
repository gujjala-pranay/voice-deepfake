"""
Train models on Balanced ElevenLabs Deepfake Dataset
"""
import os
import sys
import torch
import torch.nn as nn
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from balanced_elevenlabs_dataset import get_balanced_loaders
from src.model import get_model
from src.ensemble import create_ensemble
from src.train import EnhancedTrainer, get_default_config

def get_custom_config():
    config = {
        # Model Configuration
        'model_type': 'enhanced',  # Can be 'enhanced' or 'lightweight'
        'optimize_model': True,    # Enable model optimization (e.g., TorchScript, quantization)
        'optimization_level': 'O1', # Optimization level (e.g., 'O1' for mixed precision, 'O2' for full graph)
        'use_ensemble': False,     # Set to True to train ensemble models
        'ensemble_type': 'standard', # 'standard', 'multiscale', 'adaptive'
        
        # Training Hyperparameters
        'epochs': 100,             # Increased epochs for better convergence
        'batch_size': 32,          # Increased batch size for better GPU utilization
        'learning_rate': 0.0001,   # Further reduced learning rate for stability with larger epochs
        'weight_decay': 1e-5,      # Increased weight decay for better regularization
        'label_smoothing': 0.15,   # Slightly increased label smoothing for better calibration
        'optimizer': 'adamw',      # AdamW is generally robust
        'scheduler': 'one_cycle',  # OneCycleLR for faster convergence and better performance
        'use_amp': True,           # Automatic Mixed Precision for faster training and reduced memory usage
        'gradient_accumulation_steps': 2, # Accumulate gradients over 2 batches to simulate larger batch size
        'patience': 15,            # Increased patience for early stopping
        
        # Augmentation Configuration
        'augmentation_type': 'composite', # Use composite augmentation
        'use_batch_augmentation': True,   # Enable batch-level augmentations (Mixup/CutMix)
        'augmentation_params': {
            'audio_aug_prob': 0.6,    # Increased probability for audio augmentations
            'spectral_aug_prob': 0.4, # Increased probability for spectral augmentations
            'mixup_prob': 0.3         # Increased probability for Mixup/CutMix
        },
        
        # Dataset Configuration
        'steps_per_epoch': 0, # Will be set dynamically
    }
    return config

def train_balanced_models(data_root='./elevenlabs_dataset', epochs=None, batch_size=None):
    print("="*60)
    print("🎓 Training on Balanced ElevenLabs Deepfake Dataset")
    print("="*60)
    
    # Create data loaders
    print("\n📊 Loading dataset with balancing...")
    train_loader, val_loader, class_weights = get_balanced_loaders(
        data_root=data_root,
        batch_size=batch_size if batch_size else get_custom_config()['batch_size'],
        val_split=0.2
    )
    
    print(f"Class weights for loss: {class_weights}")
    
    # Train all model types
    models_to_train = [
        ('enhanced', 'enhanced'),
        ('lightweight', 'lightweight'),
        ('ensemble_standard', 'ensemble_standard'),
        ('ensemble_multiscale', 'ensemble_multiscale'), # Added MultiScale Ensemble
        ('ensemble_adaptive', 'ensemble_adaptive'),     # Added Adaptive Ensemble
    ]
    
    trained_models = {}
    
    for model_name, model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"🔧 Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Setup config
        config = get_custom_config()
        config['epochs'] = epochs if epochs else config['epochs']
        config['batch_size'] = batch_size if batch_size else config['batch_size']
        
        if model_type.startswith('ensemble'):
            config['use_ensemble'] = True
            config['ensemble_type'] = model_type.split('_')[1]
            # For ensemble, base models are trained first, then ensemble is created
            # For simplicity here, we'll assume base models are already trained or handled internally by create_ensemble
            # In a real scenario, you might train base models separately and then load them for ensemble training
            config['model_type'] = 'enhanced' # Ensemble models are built upon base models
        else:
            config['use_ensemble'] = False
            config['model_type'] = model_type
            
        config['steps_per_epoch'] = len(train_loader)
        
        # Create trainer
        trainer = EnhancedTrainer(config)
        
        # Use class weights in loss function to handle imbalance further
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights.to(trainer.device), 
                                               label_smoothing=config['label_smoothing'])
        
        # Train
        print(f"\n🚀 Starting training for {config['epochs']} epochs...")
        best_acc = trainer.train(train_loader, val_loader)
        
        # Save model
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f'{model_name}_balanced_elevenlabs.pth')
        torch.save(trainer.model.state_dict(), save_path)
        
        trained_models[model_name] = save_path
        print(f"✅ Model saved: {save_path}")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
    
    print("\n" + "="*60)
    print("🎉 All balanced models trained successfully!")
    print("="*60)
    
    return trained_models

def main():
    parser = argparse.ArgumentParser(description='Train on Balanced ElevenLabs Dataset')
    parser.add_argument('--data_root', type=str, default='./elevenlabs_dataset',
                       help='Path to elevenlabs_dataset folder')
    parser.add_argument('--epochs', type=int, default=None, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 3 epochs')
    
    args = parser.parse_args()
    
    epochs = 3 if args.quick_test else args.epochs
    
    train_balanced_models(
        data_root=args.data_root,
        epochs=epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
