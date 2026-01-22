#!/usr/bin/env python3
"""
EndoDetect AI - Segmentation Model Training Pipeline
Supports nnU-Net and custom Attention U-Net architectures
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# For radiomics feature extraction
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
except ImportError:
    print("Warning: PyRadiomics not installed. Radiomics features disabled.")
    RADIOMICS_AVAILABLE = False


class AttentionBlock(nn.Module):
    """Attention gate for U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net for medical image segmentation"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
            
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder with attention gates
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder with attention
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Apply attention gate
            skip_connection = self.attention_gates[idx // 2](g=x, x=skip_connection)
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
            
        return torch.sigmoid(self.final_conv(x))


class EndometriosisDataset(Dataset):
    """Dataset loader for endometriosis MRI scans"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            # Look for T2-weighted scans and corresponding masks
            images = list(subject_dir.glob("*T2*.nii*"))
            masks = list(subject_dir.glob("*mask*.nii*"))
            
            if images and masks:
                samples.append({
                    'image': images[0],
                    'mask': masks[0],
                    'subject_id': subject_dir.name
                })
                
        print(f"Found {len(samples)} training samples")
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load NIfTI files
        image = nib.load(str(sample['image'])).get_fdata()
        mask = nib.load(str(sample['mask'])).get_fdata()
        
        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Extract middle slice for 2D training
        mid_slice = image.shape[2] // 2
        image_slice = image[:, :, mid_slice]
        mask_slice = mask[:, :, mid_slice]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)
        
        return image_tensor, mask_tensor, sample['subject_id']


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice similarity coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75):
    """Focal Tversky loss for handling class imbalance"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positives, False Positives & False Negatives
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()
    
    tversky_index = TP / (TP + alpha * FP + beta * FN + 1e-6)
    focal_tversky = (1 - tversky_index) ** gamma
    
    return focal_tversky


def extract_radiomics_features(image_path, mask_path):
    """Extract radiomics features using PyRadiomics"""
    if not RADIOMICS_AVAILABLE:
        return {}
        
    try:
        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        
        # Load images
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        
        # Extract features
        features = extractor.execute(image, mask)
        
        # Filter out metadata
        radiomics_features = {
            key: float(value) for key, value in features.items() 
            if not key.startswith('diagnostics_')
        }
        
        return radiomics_features
        
    except Exception as e:
        print(f"Error extracting radiomics features: {e}")
        return {}


def train_model(model, train_loader, val_loader, device, args):
    """Training loop"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    best_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combined loss
            loss = focal_tversky_loss(outputs, masks)
            dice = dice_coefficient(outputs > 0.5, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice.item()
            
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = focal_tversky_loss(outputs, masks)
                dice = dice_coefficient(outputs > 0.5, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
        # Record history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
              
    return history, best_dice


def plot_training_history(history, output_dir):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice plot
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train EndoDetect AI segmentation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    print(f"Loading data from: {args.data_dir}")
    
    # Load dataset
    dataset = EndometriosisDataset(args.data_dir)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device(args.device)
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    history, best_dice = train_model(model, train_loader, val_loader, device, args)
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Plot results
    plot_training_history(history, args.output_dir)
    
    print(f"\n✅ Training complete! Best Dice: {best_dice:.4f}")
    print(f"Models saved to: {args.output_dir}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_dice': float(best_dice),
        'epochs': args.epochs,
        'dataset_size': len(dataset),
        'device': args.device
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
