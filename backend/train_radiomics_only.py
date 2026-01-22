#!/usr/bin/env python3
"""
EndoDetect AI - Radiomics-Only Training
Trains a model using ONLY radiomics features (no CNN)
Focuses on one dataset section (D1_MHS or D2_TCPW)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# PyRadiomics integration
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
    print("✅ PyRadiomics available")
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("❌ ERROR: PyRadiomics is required!")
    print("   Install with: pip install pyradiomics")
    print("   Note: May require Python 3.11 or earlier due to compatibility")
    sys.exit(1)


def extract_radiomics_features(image_path, mask_path=None):
    """Extract comprehensive PyRadiomics features from NIfTI image"""
    try:
        # Initialize extractor with settings optimized for endometriosis
        settings = {
            'binWidth': 25,
            'resampledPixelSpacing': None,
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True
        }
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Enable key feature classes (focus on most discriminative)
        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('Wavelet')
        
        # Enable specific feature classes
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        
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
        
        return feature_vector, feature_names
        
    except Exception as e:
        print(f"Warning: Radiomics extraction failed for {image_path}: {e}")
        return None, None


class RadiomicsOnlyDataset(Dataset):
    """Dataset that extracts and uses only radiomics features"""
    def __init__(self, patient_paths, extract_features=True):
        """
        Args:
            patient_paths: List of Path objects pointing to patient directories
            extract_features: Whether to extract radiomics features
        """
        self.patient_paths = [Path(p) for p in patient_paths]
        self.extract_features = extract_features
        self.radiomics_cache = {}
        self.feature_names = None
        self.features_list = []
        self.labels = []
        
        # Pre-extract all radiomics features
        if extract_features:
            print("Extracting radiomics features...")
            for i, patient_dir in enumerate(self.patient_paths):
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.patient_paths)} patients...")
                
                image_file, mask_file = self._find_files(patient_dir)
                if image_file is None:
                    continue
                
                # Extract radiomics
                feat, feat_names = extract_radiomics_features(image_file, mask_file)
                if feat is not None:
                    if self.feature_names is None:
                        self.feature_names = feat_names
                    self.features_list.append(feat)
                    # Label: 1 if has endometriosis mask, 0 otherwise
                    label = 1 if mask_file and mask_file.exists() else 0
                    self.labels.append(label)
            
            print(f"✅ Extracted features from {len(self.features_list)} patients")
            print(f"   Feature dimension: {len(self.feature_names) if self.feature_names else 0}")
            print(f"   Positive cases: {sum(self.labels)}")
            print(f"   Negative cases: {len(self.labels) - sum(self.labels)}")
        
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
                t2_files = [all_nii[0]]
            else:
                return None, None
        
        # Use first available endometriosis mask
        mask_file = em_masks[0] if em_masks else None
        
        return t2_files[0], mask_file
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        features = torch.from_numpy(self.features_list[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


class RadiomicsClassifier(nn.Module):
    """Simple neural network classifier using only radiomics features"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(RadiomicsClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, device, args):
    """Training loop for radiomics-only model"""
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)
    criterion = nn.BCELoss()
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = (outputs > 0.5).float()
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                pred = (outputs > 0.5).float()
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        scheduler.step()
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return history, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train EndoDetect AI with Radiomics Only')
    parser.add_argument('--data_dir', type=str, 
                       default='./data/UT-EndoMRI',
                       help='Path to UT-EndoMRI dataset')
    parser.add_argument('--dataset', type=str, choices=['D1_MHS', 'D2_TCPW'], 
                       default='D1_MHS',
                       help='Which dataset section to use')
    parser.add_argument('--output_dir', type=str, default='./models_radiomics',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(" ENDODETECT AI - RADIOMICS-ONLY TRAINING")
    print("=" * 80)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📦 Dataset section: {args.dataset}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"💻 Device: {args.device}")
    print(f"🔬 PyRadiomics: ENABLED")
    print("=" * 80 + "\n")
    
    # Load patient IDs from specified dataset section
    data_dir = Path(args.data_dir) / args.dataset
    if not data_dir.exists():
        print(f"❌ ERROR: Dataset section {args.dataset} not found in {args.data_dir}")
        sys.exit(1)
    
    patient_paths = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"📦 Found {len(patient_paths)} patients in {args.dataset}\n")
    
    # Split train/val
    train_paths, val_paths = train_test_split(patient_paths, test_size=0.2, random_state=42)
    
    print(f"📊 Split: {len(train_paths)} train, {len(val_paths)} val\n")
    
    # Create datasets (this will extract radiomics features)
    train_dataset = RadiomicsOnlyDataset(train_paths, extract_features=True)
    val_dataset = RadiomicsOnlyDataset(val_paths, extract_features=True)
    
    if len(train_dataset) == 0:
        print("❌ ERROR: No valid patients found after feature extraction")
        sys.exit(1)
    
    # Get feature dimension
    feature_dim = len(train_dataset.feature_names) if train_dataset.feature_names else train_dataset[0][0].shape[0]
    print(f"🔬 Radiomics feature dimension: {feature_dim}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device(args.device)
    model = RadiomicsClassifier(input_dim=feature_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 Model initialized: {total_params:,} parameters")
    print(f"🚀 Starting training...\n")
    
    # Train
    history, best_acc = train_model(model, train_loader, val_loader, device, args)
    
    acc_pct = best_acc * 100
    
    # Save results
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_acc'], label='Train Accuracy', linewidth=2, alpha=0.7)
    ax.plot(history['val_acc'], label='Validation Accuracy', linewidth=2.5, color='darkblue')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'EndoDetect AI - Radiomics-Only Training ({args.dataset})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    # Create metadata
    metadata = {
        'model_name': 'EndoDetect AI - Radiomics-Only Classifier',
        'training_date': datetime.now().isoformat(),
        'best_accuracy': float(best_acc),
        'best_accuracy_percentage': float(acc_pct),
        'dataset': f'UT-EndoMRI/{args.dataset}',
        'dataset_size': len(patient_paths),
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'architecture': 'Radiomics-Only Neural Network',
        'radiomics_features': feature_dim,
        'feature_names': train_dataset.feature_names if train_dataset.feature_names else [],
        'techniques': [
            'PyRadiomics feature extraction',
            'Neural network classifier',
            'Binary classification (endometriosis vs control)',
            'AdamW optimizer with cosine annealing',
            'Early stopping',
            'Gradient clipping'
        ]
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"🎯 Best Accuracy: {acc_pct:.2f}%")
    print(f"📊 Architecture: Radiomics-Only Neural Network")
    print(f"🔬 Radiomics Features: {feature_dim}")
    print(f"📦 Dataset: {args.dataset}")
    print(f"💾 Saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
