"""
Train models on ElevenLabs Deepfake Dataset
"""
import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from elevenlabs_dataset import get_elevenlabs_loaders
from src.model import get_model
from src.ensemble import create_ensemble
from src.train import EnhancedTrainer, get_default_config

def train_elevenlabs_models(data_root='./elevenlabs_dataset', epochs=50, batch_size=16):
    """
    Train all models (enhanced, lightweight, ensemble) on ElevenLabs dataset
    """
    print("="*60)
    print("🎓 Training on ElevenLabs Deepfake Dataset")
    print("="*60)
    
    # Create data loaders
    print("\n📊 Loading dataset...")
    train_loader, val_loader = get_elevenlabs_loaders(
        data_root=data_root,
        batch_size=batch_size,
        val_split=0.2
    )
    
    # Train all model types
    models_to_train = [
        ('enhanced', 'enhanced'),
        ('lightweight', 'lightweight'),
        ('ensemble_standard', 'ensemble_standard'),
    ]
    
    trained_models = {}
    
    for model_name, model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"🔧 Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Setup config
        config = get_default_config()
        config['model_type'] = model_type
        config['epochs'] = epochs
        config['batch_size'] = batch_size
        config['use_ensemble'] = model_type.startswith('ensemble')
        config['ensemble_type'] = model_type.split('_')[1] if '_' in model_type else 'standard'
        config['steps_per_epoch'] = len(train_loader)
        config['learning_rate'] = 0.001
        config['patience'] = 10
        
        # Create trainer
        trainer = EnhancedTrainer(config)
        
        # Train
        print(f"\n🚀 Starting training for {epochs} epochs...")
        best_acc = trainer.train(train_loader, val_loader)
        
        # Save model
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f'{model_name}_elevenlabs.pth')
        torch.save(trainer.model.state_dict(), save_path)
        
        trained_models[model_name] = save_path
        print(f"✅ Model saved: {save_path}")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
    
    print("\n" + "="*60)
    print("🎉 All models trained successfully!")
    print("="*60)
    print("\nTrained models:")
    for name, path in trained_models.items():
        print(f"   • {name}: {path}")
    
    print("\n📋 Next steps:")
    print("   1. Run prediction: streamlit run app.py")
    print("   2. Or: python predict.py --audio <file.wav>")
    
    return trained_models

def quick_train_test(data_root='./elevenlabs_dataset', epochs=3):
    """Quick training test with minimal epochs"""
    print("\n🧪 Quick Training Test (3 epochs)...")
    return train_elevenlabs_models(data_root=data_root, epochs=epochs, batch_size=8)

def main():
    parser = argparse.ArgumentParser(description='Train on ElevenLabs Dataset')
    parser.add_argument('--data_root', type=str, default='./elevenlabs_dataset',
                       help='Path to elevenlabs_dataset folder')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 3 epochs')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'enhanced', 'lightweight', 'ensemble'],
                       help='Which model to train')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_train_test(args.data_root)
    else:
        train_elevenlabs_models(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
