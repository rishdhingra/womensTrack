#!/usr/bin/env python3
"""
EndoDetect AI - Radiomics-Based Training Pipeline
Focus: Train model using radiomics features from UT-EndoMRI dataset
Removed: All demo generation functionality
"""

import os
import sys
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
from scipy.ndimage import rotate, zoom, gaussian_filter
from sklearn.model_selection import train_test_split

# PyRadiomics integration - Optional for now due to Python 3.12 compatibility issues
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
    print("✅ PyRadiomics available")
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("⚠️  WARNING: PyRadiomics not available - training without radiomics features")
    print("   The model will use CNN features only")
    print("   To install PyRadiomics later: pip install pyradiomics")

# Import model architecture
sys.path.append(str(Path(__file__).parent))
from train_enhanced import AttentionUNet, dice_coefficient


class RadiomicsEnhancedUNet(nn.Module):
    """U-Net enhanced with radiomics feature branch"""
    def __init__(self, in_channels=1, out_channels=1, radiomics_dim=100):
        super(RadiomicsEnhancedUNet, self).__init__()
        self.unet = AttentionUNet(in_channels, out_channels)
        
        # Radiomics feature processor
        self.radiomics_fc = nn.Sequential(
            nn.Linear(radiomics_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Feature fusion layer
        self.fusion = nn.Conv2d(65, 1, kernel_size=1)  # 64 from radiomics + 1 from UNet
        
    def forward(self, x, radiomics_features=None):
        # U-Net prediction
        unet_out = self.unet(x)
        
        if radiomics_features is not None:
            # Process radiomics features
            radio_feat = self.radiomics_fc(radiomics_features)
            
            # Broadcast to spatial dimensions
            b, c = radio_feat.shape
            h, w = unet_out.shape[2], unet_out.shape[3]
            radio_spatial = radio_feat.view(b, c, 1, 1).expand(b, c, h, w)
            
            # Fuse with U-Net output
            combined = torch.cat([unet_out, radio_spatial], dim=1)
            output = torch.sigmoid(self.fusion(combined))
        else:
            output = unet_out
        
        return output


def extract_radiomics_features(image_path, mask_path=None):
    """Extract comprehensive PyRadiomics features from NIfTI image"""
    if not RADIOMICS_AVAILABLE:
        return None, None
    try:
        # Initialize extractor with settings optimized for endometriosis
        settings = {
            'binWidth': 25,
            'resampledPixelSpacing': None,
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True
        }
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Enable all image types for comprehensive feature extraction
        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('LoG')
        extractor.enableImageTypeByName('Square')
        extractor.enableImageTypeByName('SquareRoot')
        extractor.enableImageTypeByName('Logarithm')
        extractor.enableImageTypeByName('Exponential')
        extractor.enableImageTypeByName('Gradient')
        extractor.enableImageTypeByName('LBP2D')
        
        # Enable all feature classes
        extractor.enableAllFeatures()
        
        # Load image
        image = sitk.ReadImage(str(image_path))
        
        # Create or load mask
        if mask_path and Path(mask_path).exists():
            mask = sitk.ReadImage(str(mask_path))
        else:
            # Create whole-image mask if no mask provided
            arr = sitk.GetArrayFromImage(image)
            mask_arr = (arr > np.percentile(arr, 10)).astype(np.uint8)
            mask = sitk.GetImageFromArray(mask_arr)
            mask.CopyInformation(image)
        
        # Extract features
        result = extractor.execute(image, mask)
        
        # Filter numeric features (exclude diagnostics)
        features = {}
        for key, val in result.items():
            if not key.startswith('diagnostics_'):
                try:
                    features[key] = float(val)
                except (ValueError, TypeError):
                    pass
        
        # Convert to feature vector (sorted for consistency)
        feature_names = sorted(features.keys())
        feature_vector = np.array([features.get(name, 0.0) for name in feature_names])
        
        # Handle NaN and Inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        if feature_vector.std() > 1e-8:
            feature_vector = (feature_vector - feature_vector.mean()) / (feature_vector.std() + 1e-8)
        
        return feature_vector, feature_names
        
    except Exception as e:
        print(f"Warning: Radiomics extraction failed for {image_path}: {e}")
        return None, None


class UTEndoMRIDataset(Dataset):
    """Dataset loader for UT-EndoMRI dataset with radiomics integration"""
    def __init__(self, patient_paths, augment=False, extract_radiomics=True):
        """
        Args:
            patient_paths: List of Path objects pointing to patient directories
        """
        self.patient_paths = [Path(p) for p in patient_paths]
        self.augment = augment
        self.extract_radiomics = extract_radiomics
        self.radiomics_cache = {}
        self.feature_names = None
        
    def __len__(self):
        return len(self.patient_paths) * (20 if self.augment else 1)
        
    def _find_files(self, patient_dir):
        """Find T2 image and endometriosis mask files"""
        # Find T2 image - try multiple patterns
        t2_files = (list(patient_dir.glob("*_T2.nii.gz")) or 
                    list(patient_dir.glob("*T2*.nii.gz")) or
                    list(patient_dir.glob("*T2*.nii")) or
                    list(patient_dir.glob("*t2*.nii.gz")))
        
        # Find endometriosis masks (em_r1, em_r2, em_r3)
        em_masks = sorted(patient_dir.glob("*_em_r*.nii.gz"))
        
        if not t2_files:
            # Try to find any NIfTI file as fallback
            all_nii = list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.nii"))
            if all_nii:
                t2_files = [all_nii[0]]  # Use first NIfTI file found
            else:
                return None, None
        
        # Use first available endometriosis mask, or create empty mask
        mask_file = em_masks[0] if em_masks else None
        
        return t2_files[0], mask_file
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.patient_paths)
        patient_dir = self.patient_paths[real_idx]
        patient_id = patient_dir.name
        
        # Find T2 image and mask
        image_file, mask_file = self._find_files(patient_dir)
        
        if image_file is None:
            raise ValueError(f"Missing T2 image for {patient_id}")
        
        # Load data
        image = nib.load(str(image_file)).get_fdata()
        
        if mask_file and mask_file.exists():
            mask = nib.load(str(mask_file)).get_fdata()
        else:
            # Create empty mask if no mask file
            mask = np.zeros_like(image)
        
        # Extract middle slice for 2D training
        mid_slice = image.shape[2] // 2
        image_slice = image[:, :, mid_slice].copy()
        mask_slice = mask[:, :, mid_slice].copy()
        
        # Resize to exactly 256x256 using a more robust method
        target_size = (256, 256)
        if image_slice.shape != target_size:
            from scipy.ndimage import zoom
            # Calculate zoom factors
            zoom_y = target_size[0] / image_slice.shape[0]
            zoom_x = target_size[1] / image_slice.shape[1]
            # Apply zoom
            image_slice = zoom(image_slice, (zoom_y, zoom_x), order=1, mode='nearest')
            mask_slice = zoom(mask_slice, (zoom_y, zoom_x), order=0, mode='nearest')
            # Force exact size by cropping or padding - use _resize_to_256 helper
            image_slice = self._resize_to_256(image_slice)
            mask_slice = self._resize_to_256(mask_slice)
        
        # Ensure exact 256x256 before augmentation
        if image_slice.shape != (256, 256):
            image_slice = self._resize_to_256(image_slice)
        if mask_slice.shape != (256, 256):
            mask_slice = self._resize_to_256(mask_slice)
        
        # Augmentation for training
        if self.augment:
            image_slice, mask_slice = self._augment(image_slice, mask_slice)
            # Ensure still 256x256 after augmentation
            if image_slice.shape != (256, 256):
                image_slice = self._resize_to_256(image_slice)
            if mask_slice.shape != (256, 256):
                mask_slice = self._resize_to_256(mask_slice)
        
        # Ensure contiguous
        image_slice = np.ascontiguousarray(image_slice)
        mask_slice = np.ascontiguousarray(mask_slice)
        
        # Normalize
        image_slice = (image_slice - image_slice.mean()) / (image_slice.std() + 1e-8)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice > 0).float().unsqueeze(0)
        
        # Extract radiomics features
        radiomics_features = None
        if self.extract_radiomics:
            cache_key = f"{patient_id}_{image_file.name}"
            if cache_key not in self.radiomics_cache:
                feat, feat_names = extract_radiomics_features(image_file, mask_file)
                if feat is not None:
                    self.radiomics_cache[cache_key] = feat
                    if self.feature_names is None:
                        self.feature_names = feat_names
            radiomics_features = self.radiomics_cache.get(cache_key)
            if radiomics_features is not None:
                radiomics_features = torch.from_numpy(radiomics_features).float()
            else:
                # Create zero features if extraction failed
                if self.feature_names:
                    radiomics_features = torch.zeros(len(self.feature_names))
                else:
                    radiomics_features = torch.zeros(100)  # Default size
        
        return image_tensor, mask_tensor, radiomics_features, patient_id
    
    def _augment(self, image, mask):
        """Heavy augmentation pipeline"""
        # Rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-25, 25)
            image = rotate(image, angle, reshape=False, order=1)
            mask = rotate(mask, angle, reshape=False, order=0)
        
        # Zoom
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            image = zoom(image, scale, order=1)
            mask = zoom(mask, scale, order=0)
            # Crop/pad to 256x256
            image = self._resize_to_256(image)
            mask = self._resize_to_256(mask)
        
        # Flips
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if np.random.random() > 0.3:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Intensity augmentations
        if np.random.random() > 0.5:
            image = image * np.random.uniform(0.8, 1.2)
        if np.random.random() > 0.5:
            image = gaussian_filter(image, sigma=np.random.uniform(0.5, 1.5))
        if np.random.random() > 0.5:
            image = image + np.random.randn(*image.shape) * 0.05
        
        return image, mask
    
    def _resize_to_256(self, arr):
        """Resize array to exactly 256x256"""
        target_size = (256, 256)
        if arr.shape == target_size:
            return arr.copy()
        
        from scipy.ndimage import zoom
        # Calculate zoom factors
        zoom_y = target_size[0] / arr.shape[0]
        zoom_x = target_size[1] / arr.shape[1]
        
        # Apply zoom
        resized = zoom(arr, (zoom_y, zoom_x), order=1 if arr.dtype == np.float64 else 0, mode='nearest')
        
        # Create output array with exact target size
        result = np.zeros(target_size, dtype=arr.dtype)
        
        # Copy resized data (handle both too large and too small cases)
        copy_h = min(resized.shape[0], target_size[0])
        copy_w = min(resized.shape[1], target_size[1])
        result[:copy_h, :copy_w] = resized[:copy_h, :copy_w]
        
        return result


def combined_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, dice_weight=0.5):
    """Advanced combined loss: Focal Tversky + Dice + BCE"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Focal Tversky
    TP = (pred_flat * target_flat).sum()
    FP = ((1 - target_flat) * pred_flat).sum()
    FN = (target_flat * (1 - pred_flat)).sum()
    tversky_index = TP / (TP + alpha * FP + beta * FN + 1e-6)
    focal_tversky = (1 - tversky_index) ** gamma
    
    # Dice loss
    dice_loss = 1 - dice_coefficient(pred, target)
    
    # BCE loss
    bce = nn.BCELoss()(pred_flat, target_flat)
    
    # Combine
    return 0.5 * focal_tversky + dice_weight * dice_loss + 0.2 * bce


def train_model(model, train_loader, val_loader, device, args):
    """Training loop with radiomics integration"""
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)
    
    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_dice = 0.0, 0.0
        
        for batch in train_loader:
            images, masks, radiomics, _ = batch
            images = images.to(device)
            masks = masks.to(device)
            radiomics = radiomics.to(device) if radiomics is not None else None
            
            optimizer.zero_grad()
            outputs = model(images, radiomics)
            
            loss = combined_loss(outputs, masks)
            dice = dice_coefficient(outputs > 0.5, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice.item()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images, masks, radiomics, _ = batch
                images = images.to(device)
                masks = masks.to(device)
                radiomics = radiomics.to(device) if radiomics is not None else None
                
                outputs = model(images, radiomics)
                loss = combined_loss(outputs, masks)
                dice = dice_coefficient(outputs > 0.5, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        scheduler.step()
        
        # Save best
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
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
    
    return history, best_dice


def main():
    parser = argparse.ArgumentParser(description='Train EndoDetect AI with Radiomics on UT-EndoMRI')
    parser.add_argument('--data_dir', type=str, 
                       default='./data/UT-EndoMRI',
                       help='Path to UT-EndoMRI dataset (should contain D1_MHS and D2_TCPW)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--use_d1', action='store_true',
                       help='Use D1_MHS dataset')
    parser.add_argument('--use_d2', action='store_true',
                       help='Use D2_TCPW dataset')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(" ENDODETECT AI - RADIOMICS-BASED TRAINING")
    print("=" * 80)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"💻 Device: {args.device}")
    print(f"🔬 PyRadiomics: {'ENABLED' if RADIOMICS_AVAILABLE else 'DISABLED (using CNN only)'}")
    print("=" * 80 + "\n")
    
    # Load patient IDs from UT-EndoMRI dataset
    data_dir = Path(args.data_dir)
    all_patients = []
    
    # Load from D1_MHS if specified or if no specific dataset chosen
    if args.use_d1 or (not args.use_d1 and not args.use_d2):
        d1_dir = data_dir / 'D1_MHS'
        if d1_dir.exists():
            d1_patients = [d.name for d in d1_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            all_patients.extend([(d1_dir / p) for p in d1_patients])
            print(f"📦 D1_MHS: {len(d1_patients)} patients")
    
    # Load from D2_TCPW if specified
    if args.use_d2 or (not args.use_d1 and not args.use_d2):
        d2_dir = data_dir / 'D2_TCPW'
        if d2_dir.exists():
            d2_patients = [d.name for d in d2_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            all_patients.extend([(d2_dir / p) for p in d2_patients])
            print(f"📦 D2_TCPW: {len(d2_patients)} patients")
    
    if not all_patients:
        print(f"❌ ERROR: No patients found in {args.data_dir}")
        print("Make sure UT-EndoMRI dataset is in the data directory")
        sys.exit(1)
    
    # Convert to patient IDs (just the folder name)
    patient_ids = [p.name for p in all_patients]
    patient_dirs = {p.name: p for p in all_patients}
    
    # Split train/val
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    
    print(f"📊 Total: {len(patient_ids)} patients ({len(train_ids)} train, {len(val_ids)} val)\n")
    
    # Create patient path lists for train and validation
    train_paths = [patient_dirs[pid] for pid in train_ids if pid in patient_dirs]
    val_paths = [patient_dirs[pid] for pid in val_ids if pid in patient_dirs]
    
    train_dataset = UTEndoMRIDataset(train_paths, augment=True, extract_radiomics=True)
    val_dataset = UTEndoMRIDataset(val_paths, augment=False, extract_radiomics=True)
    
    # Get radiomics feature dimension from first sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        radiomics_dim = sample[2].shape[0] if sample[2] is not None else 100
    else:
        radiomics_dim = 100
    
    print(f"🔬 Radiomics feature dimension: {radiomics_dim}\n")
    
    # Custom collate function
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        radiomics = [item[2] for item in batch]
        if radiomics[0] is not None:
            radiomics = torch.stack(radiomics)
        else:
            radiomics = None
        ids = [item[3] for item in batch]
        return images, masks, radiomics, ids
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=0, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device(args.device)
    model = RadiomicsEnhancedUNet(radiomics_dim=radiomics_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Model initialized: {total_params:,} parameters")
    print(f"🚀 Starting training...\n")
    
    # Train
    history, best_dice = train_model(model, train_loader, val_loader, device, args)
    
    dice_pct = best_dice * 100
    
    # Save results
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_dice'], label='Train Dice', linewidth=2, alpha=0.7)
    ax.plot(history['val_dice'], label='Validation Dice', linewidth=2.5, color='darkblue')
    ax.axhline(y=0.75, color='green', linestyle='--', label='Target (75%)', linewidth=2)
    ax.axhline(y=0.82, color='orange', linestyle=':', label='UT-EndoMRI Benchmark (82%)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Dice Coefficient', fontsize=14)
    ax.set_title('EndoDetect AI - Radiomics Training Performance', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    # Create metadata
    metadata = {
        'model_name': 'EndoDetect AI - Radiomics-Enhanced Model',
        'training_date': datetime.now().isoformat(),
        'best_dice_score': float(best_dice),
        'best_dice_percentage': float(dice_pct),
        'dataset': 'UT-EndoMRI',
        'dataset_size': len(patient_ids),
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'architecture': 'Radiomics-Enhanced Attention U-Net',
        'radiomics_features': radiomics_dim,
        'techniques': [
            'PyRadiomics feature extraction',
            'Attention U-Net segmentation',
            'Radiomics-CNN feature fusion',
            'Heavy data augmentation',
            'Focal Tversky + Dice + BCE loss',
            'AdamW optimizer with cosine annealing',
            'Early stopping',
            'Gradient clipping'
        ],
        'target_performance': '75-85% Dice (literature benchmark)',
        'citation': 'Liang et al. 2025 (UT-EndoMRI) - DOI: 10.5281/zenodo.15750762'
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"🎯 Best Dice Score: {dice_pct:.2f}%")
    print(f"📊 Architecture: Radiomics-Enhanced Attention U-Net")
    print(f"🔬 Radiomics Features: {radiomics_dim}")
    print(f"💾 Saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
