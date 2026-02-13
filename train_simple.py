"""
Simple training script for ElevenLabs dataset
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from elevenlabs_dataset import ElevenLabsDataset
from src.model import get_model

def simple_train(epochs=5, batch_size=8):
    print("="*60)
    print("🎓 Simple Training on ElevenLabs Dataset")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = ElevenLabsDataset('./elevenlabs_dataset', augment=True)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    val_dataset.dataset.augment = False
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2,pin_memory=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    model = get_model('enhanced')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            spectral = batch['spectral'].to(device)
            mfcc = batch['mfcc'].to(device)
            phase = batch['phase'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(spectral, mfcc, phase)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                spectral = batch['spectral'].to(device)
                mfcc = batch['mfcc'].to(device)
                phase = batch['phase'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(spectral, mfcc, phase)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/enhanced_elevenlabs.pth')
            print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Training complete! Best Val Acc: {best_acc:.2f}%")
    print("📁 Model saved: model/enhanced_elevenlabs.pth")

if __name__ == "__main__":
    simple_train()
