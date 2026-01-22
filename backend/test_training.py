#!/usr/bin/env python3
"""
Quick training test - trains model for 1 epoch on dummy data
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from train_segmentation_model import AttentionUNet, dice_coefficient, focal_tversky_loss

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self, num_samples=10, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random MRI-like image
        image = torch.randn(1, self.img_size, self.img_size).float()
        # Create random mask (binary segmentation)
        mask = (torch.rand(1, self.img_size, self.img_size) > 0.7).float()
        return image, mask, f"sample_{idx}"

def test_training():
    """Test that training loop works"""
    print("="*60)
    print("TRAINING TEST - 1 Epoch on Dummy Data")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create model
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy dataset
    dataset = DummyDataset(num_samples=5, img_size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = focal_tversky_loss
    
    model.train()
    print("\nStarting training...")
    
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        dice = dice_coefficient(outputs > 0.5, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss={loss.item():.4f}, Dice={dice.item():.4f}")
    
    print("\n✅ Training test completed successfully!")
    print(f"   Final loss: {loss.item():.4f}")
    print(f"   Final Dice: {dice.item():.4f}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        test_image = torch.randn(1, 1, 256, 256).to(device)
        prediction = model(test_image)
        print(f"\n✅ Inference test: Output shape {prediction.shape}")
        print(f"   Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    
    return model

if __name__ == "__main__":
    model = test_training()
    print("\n" + "="*60)
    print("✅ Model training functionality verified!")
    print("="*60)
