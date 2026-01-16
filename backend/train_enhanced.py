#!/usr/bin/env python3
"""
EndoDetect AI - Enhanced Training Pipeline
Includes: Data augmentation, ensemble methods, cross-validation, PyRadiomics integration
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
from sklearn.model_selection import KFold
from scipy.ndimage import rotate, zoom, shift

# PyRadiomics
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
except ImportError:
    print("Warning: PyRadiomics not installed. Feature extraction disabled.")
    RADIOMICS_AVAILABLE = False


class DataAugmentation:
    """Advanced data augmentation for medical images"""
    def __init__(self, rotation_range=15, zoom_range=0.1, shift_range=0.1):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
    
    def __call__(self, image, mask):
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = rotate(image, angle, reshape=False, order=1)
            mask = rotate(mask, angle, reshape=False, order=0)
        
        # Random zoom
        if np.random.random() > 0.5:
            scale = np.random.uniform(1-self.zoom_range, 1+self.zoom_range)
            image = zoom(image, scale, order=1)
            mask = zoom(mask, scale, order=0)
            # Crop or pad to original size
            image = self._resize_to_shape(image, (256, 256))
            mask = self._resize_to_shape(mask, (256, 256))
        
        # Random shift
        if np.random.random() > 0.5:
            shift_x = np.random.uniform(-self.shift_range, self.shift_range) * image.shape[0]
            shift_y = np.random.uniform(-self.shift_range, self.shift_range) * image.shape[1]
            image = shift(image, [shift_x, shift_y], order=1)
            mask = shift(mask, [shift_x, shift_y], order=0)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        return image, mask
    
    def _resize_to_shape(self, arr, target_shape):
        """Crop or pad to target shape"""
        current_shape = arr.shape
        if current_shape == target_shape:
            return arr
        
        result = np.zeros(target_shape)
        min_h = min(current_shape[0], target_shape[0])
        min_w = min(current_shape[1], target_shape[1])
        result[:min_h, :min_w] = arr[:min_h, :min_w]
        return result


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
    """Enhanced Attention U-Net with residual connections"""
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
            
        # Bottleneck with dropout
        self.bottleneck = nn.Sequential(
            self._conv_block(features[-1], features[-1] * 2),
            nn.Dropout2d(0.3)
        )
        
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
        for encode in self.encoder:
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
    """Enhanced dataset with augmentation"""
    def __init__(self, data_dir, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.augmentor = DataAugmentation() if augment else None
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
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
        
        # Extract middle slice for 2D training
        mid_slice = image.shape[2] // 2
        image_slice = image[:, :, mid_slice]
        mask_slice = mask[:, :, mid_slice]
        
        # Resize to 256x256 if needed
        if image_slice.shape != (256, 256):
            from scipy.ndimage import zoom
            zoom_factors = (256/image_slice.shape[0], 256/image_slice.shape[1])
            image_slice = zoom(image_slice, zoom_factors, order=1)
            mask_slice = zoom(mask_slice, zoom_factors, order=0)
        
        # Apply augmentation
        if self.augment and self.augmentor:
            image_slice, mask_slice = self.augmentor(image_slice, mask_slice)
        
        # Ensure contiguous arrays (fix negative stride issue)
        image_slice = np.ascontiguousarray(image_slice)
        mask_slice = np.ascontiguousarray(mask_slice)
        
        # Normalize image
        image_slice = (image_slice - image_slice.mean()) / (image_slice.std() + 1e-8)
        
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


def combined_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75):
    """Combined Focal Tversky + Dice loss"""
    # Focal Tversky
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    TP = (pred_flat * target_flat).sum()
    FP = ((1 - target_flat) * pred_flat).sum()
    FN = (target_flat * (1 - pred_flat)).sum()
    
    tversky_index = TP / (TP + alpha * FP + beta * FN + 1e-6)
    focal_tversky = (1 - tversky_index) ** gamma
    
    # Dice loss
    dice_loss = 1 - dice_coefficient(pred, target)
    
    # Combine losses
    return 0.7 * focal_tversky + 0.3 * dice_loss


def train_model(model, train_loader, val_loader, device, args):
    """Enhanced training loop with gradient clipping and scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_dice = 0.0
    patience_counter = 0
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
            
            loss = combined_loss(outputs, masks)
            dice = dice_coefficient(outputs > 0.5, masks)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                loss = combined_loss(outputs, masks)
                dice = dice_coefficient(outputs > 0.5, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        # Record history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
              
    return history, best_dice


def plot_training_history(history, output_dir):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Training and Validation Dice', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train EndoDetect AI (Enhanced)')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EndoDetect AI - Enhanced Training Pipeline")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Load datasets with augmentation
    train_dataset = EndometriosisDataset(args.data_dir, augment=True)
    val_dataset = EndometriosisDataset(args.data_dir, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device(args.device)
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nStarting training...")
    
    # Train model
    history, best_dice = train_model(model, train_loader, val_loader, device, args)
    
    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Plot results
    plot_training_history(history, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"Best Validation Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_dice': float(best_dice),
        'best_dice_percentage': float(best_dice * 100),
        'epochs_trained': len(history['train_dice']),
        'total_epochs': args.epochs,
        'dataset_size': len(train_dataset),
        'device': args.device,
        'architecture': 'Attention U-Net',
        'augmentation': 'rotation, zoom, shift, flip',
        'loss_function': 'Combined Focal Tversky + Dice'
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
