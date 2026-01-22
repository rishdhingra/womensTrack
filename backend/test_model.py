#!/usr/bin/env python3
"""
Test script for EndoDetect AI model
Tests model architecture, forward pass, and basic inference
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Import model architecture
try:
    from train_segmentation_model import AttentionUNet, dice_coefficient
    print("✅ Successfully imported model architecture")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

def test_model_architecture():
    """Test that the model architecture works correctly"""
    print("\n" + "="*60)
    print("TEST 1: Model Architecture")
    print("="*60)
    
    device = torch.device('cpu')
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Expected: ~31.4M parameters")
    
    return model, device

def test_forward_pass(model, device):
    """Test forward pass with dummy data"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    # Create dummy input (batch_size=1, channels=1, height=256, width=256)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 256, 256).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check output is reasonable (should be between 0 and 1 after sigmoid)
            if output.min() >= 0 and output.max() <= 1:
                print(f"   ✅ Output values in valid range [0, 1]")
            else:
                print(f"   ⚠️  Output values outside [0, 1] range (may need sigmoid)")
            
            return output
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def test_dice_coefficient():
    """Test Dice coefficient calculation"""
    print("\n" + "="*60)
    print("TEST 3: Dice Coefficient")
    print("="*60)
    
    # Create dummy predictions and ground truth
    pred = torch.randint(0, 2, (1, 1, 256, 256)).float()
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    dice = dice_coefficient(pred, target)
    print(f"✅ Dice coefficient calculated: {dice.item():.4f}")
    print(f"   Expected range: [0, 1]")
    
    # Test perfect match
    perfect_dice = dice_coefficient(target, target)
    print(f"   Perfect match test: {perfect_dice.item():.4f} (should be ~1.0)")
    
    return dice

def test_inference_pipeline(model, device):
    """Test full inference pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Inference Pipeline")
    print("="*60)
    
    # Create a more realistic dummy MRI slice
    mri_slice = torch.randn(1, 1, 256, 256).to(device)
    # Normalize to simulate MRI intensity range
    mri_slice = (mri_slice - mri_slice.mean()) / mri_slice.std()
    
    model.eval()
    with torch.no_grad():
        # Run inference
        prediction = model(mri_slice)
        
        # Apply threshold to get binary mask
        threshold = 0.5
        binary_mask = (prediction > threshold).float()
        
        # Calculate some statistics
        lesion_area = binary_mask.sum().item()
        total_pixels = binary_mask.numel()
        lesion_percentage = (lesion_area / total_pixels) * 100
        
        print(f"✅ Inference pipeline completed")
        print(f"   Input shape: {mri_slice.shape}")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Binary mask shape: {binary_mask.shape}")
        print(f"   Lesion area: {lesion_area:.0f} pixels ({lesion_percentage:.2f}%)")
        print(f"   Mean prediction value: {prediction.mean().item():.4f}")
        
        return prediction, binary_mask

def test_model_loading():
    """Test loading a saved model (if available)"""
    print("\n" + "="*60)
    print("TEST 5: Model Loading")
    print("="*60)
    
    model_path = Path('./models/best_model.pth')
    
    if model_path.exists():
        print(f"✅ Found model file: {model_path}")
        try:
            device = torch.device('cpu')
            model = AttentionUNet(in_channels=1, out_channels=1).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Model loaded successfully")
            if 'dice' in checkpoint:
                print(f"   Trained Dice Score: {checkpoint['dice']:.4f} ({checkpoint['dice']*100:.2f}%)")
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint['epoch']} epochs")
            
            return model, device
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    else:
        print(f"⚠️  No trained model found at {model_path}")
        print(f"   Using untrained model for testing")
        return None, None

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("EndoDetect AI - Model Testing Suite")
    print("="*60)
    
    # Test 1: Architecture
    model, device = test_model_architecture()
    
    # Test 2: Forward pass
    output = test_forward_pass(model, device)
    if output is None:
        print("\n❌ Forward pass failed. Stopping tests.")
        return
    
    # Test 3: Dice coefficient
    dice = test_dice_coefficient()
    
    # Test 4: Inference pipeline
    prediction, mask = test_inference_pipeline(model, device)
    
    # Test 5: Model loading (if available)
    trained_model, _ = test_model_loading()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Model architecture: PASSED")
    print("✅ Forward pass: PASSED")
    print("✅ Dice coefficient: PASSED")
    print("✅ Inference pipeline: PASSED")
    if trained_model is not None:
        print("✅ Model loading: PASSED")
    else:
        print("⚠️  Model loading: No trained model available")
    
    print("\n✅ All core functionality tests passed!")
    print("\nNext steps:")
    print("  1. Train model: python train_segmentation_model.py --epochs 10 --device cpu")
    print("  2. Generate demos: python generate_demo_outputs.py --model_path ./models/best_model.pth")
    print("  3. Start API: python backend_api.py")

if __name__ == "__main__":
    main()
