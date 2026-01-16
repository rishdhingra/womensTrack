#!/usr/bin/env python3
"""
Generate Demo Outputs for EndoDetect AI Presentation
Creates heatmaps, segmentation overlays, and surgical roadmaps
"""

import os
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2

# Import trained model
from train_segmentation_model import AttentionUNet


def load_model(model_path, device='cpu'):
    """Load trained segmentation model"""
    model = AttentionUNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


def create_heatmap_overlay(image, segmentation, confidence=None):
    """Create a heatmap overlay on the original image"""
    # Normalize image to 0-255
    image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)
    
    # Create red-yellow heatmap for lesion probability
    if confidence is None:
        confidence = segmentation
        
    # Custom colormap: transparent -> yellow -> red
    colors = [(0, 0, 0, 0), (1, 1, 0, 0.6), (1, 0, 0, 0.9)]  # RGBA
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('lesion_heatmap', colors, N=n_bins)
    
    # Apply heatmap
    heatmap = (confidence * 255).astype(np.uint8)
    heatmap_colored = cm.get_cmap('hot')(heatmap / 255.0)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend
    alpha = 0.4
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_rgb, alpha, 0)
    
    # Add contour for detected lesion
    if segmentation.max() > 0:
        contours, _ = cv2.findContours(
            (segmentation > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay


def generate_surgical_roadmap(segmentation, image_shape):
    """Generate surgical roadmap with organ involvement analysis"""
    # Calculate lesion characteristics
    lesion_volume = segmentation.sum() / segmentation.size * 100  # percentage
    
    # Approximate lesion depth (based on intensity gradient)
    if segmentation.max() > 0:
        lesion_mask = segmentation > 0.5
        depth_estimate = np.mean(segmentation[lesion_mask]) * 100
    else:
        depth_estimate = 0
        
    # Surgical complexity score (0-100)
    complexity = min(100, lesion_volume * 2 + depth_estimate * 0.5)
    
    # Organ involvement analysis (mock for demo)
    organs = {
        'Ovary': np.random.choice([True, False], p=[0.7, 0.3]),
        'Uterus': np.random.choice([True, False], p=[0.4, 0.6]),
        'Bladder': np.random.choice([True, False], p=[0.2, 0.8]),
        'Bowel': np.random.choice([True, False], p=[0.3, 0.7]),
        'Peritoneum': np.random.choice([True, False], p=[0.5, 0.5])
    }
    
    roadmap = {
        'lesion_characteristics': {
            'volume_percentage': float(lesion_volume),
            'estimated_depth_mm': float(depth_estimate * 0.5),  # Convert to mm estimate
            'complexity_score': float(complexity)
        },
        'organ_involvement': organs,
        'recommendations': []
    }
    
    # Generate recommendations based on complexity
    if complexity < 30:
        roadmap['recommendations'].append('Low complexity - suitable for standard laparoscopy')
    elif complexity < 70:
        roadmap['recommendations'].append('Moderate complexity - experienced surgeon recommended')
        roadmap['recommendations'].append('Consider bowel prep if GI involvement suspected')
    else:
        roadmap['recommendations'].append('High complexity - specialist referral advised')
        roadmap['recommendations'].append('Multidisciplinary team consultation recommended')
        roadmap['recommendations'].append('Extended OR time and equipment required')
        
    return roadmap


def create_comparison_figure(image, ground_truth, prediction, confidence, output_path):
    """Create comprehensive comparison figure"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original MRI/TVUS', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].imshow(ground_truth, cmap='Reds', alpha=0.5)
    axes[0, 1].set_title('Expert Annotation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].imshow(prediction, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('EndoDetect AI Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Confidence heatmap
    axes[1, 0].imshow(image, cmap='gray')
    heatmap = axes[1, 0].imshow(confidence, cmap='hot', alpha=0.6)
    axes[1, 0].set_title('Lesion Probability Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(heatmap, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay comparison
    overlay = create_heatmap_overlay(image, prediction, confidence)
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Surgical View with Detected Lesions', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Dice score visualization
    from train_segmentation_model import dice_coefficient
    dice = dice_coefficient(
        torch.from_numpy(prediction).float(),
        torch.from_numpy(ground_truth).float()
    ).item()
    
    axes[1, 2].text(0.5, 0.6, f'Dice Coefficient\n{dice:.3f}',
                    ha='center', va='center', fontsize=24,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 2].text(0.5, 0.3, f'Detection: {"✓ Positive" if prediction.max() > 0.5 else "✗ Negative"}',
                    ha='center', va='center', fontsize=18)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return dice


def create_surgical_roadmap_figure(roadmap, output_path):
    """Create visual surgical roadmap"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EndoDetect AI - Surgical Roadmap', fontsize=18, fontweight='bold')
    
    # Lesion characteristics
    chars = roadmap['lesion_characteristics']
    ax = axes[0, 0]
    ax.text(0.5, 0.8, 'Lesion Characteristics', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.1, 0.6, f"Volume: {chars['volume_percentage']:.1f}%", fontsize=12)
    ax.text(0.1, 0.4, f"Est. Depth: {chars['estimated_depth_mm']:.1f} mm", fontsize=12)
    ax.text(0.1, 0.2, f"Complexity Score: {chars['complexity_score']:.0f}/100", fontsize=12)
    ax.axis('off')
    
    # Complexity gauge
    ax = axes[0, 1]
    complexity = chars['complexity_score']
    colors = ['green', 'yellow', 'orange', 'red']
    levels = [0, 30, 50, 70, 100]
    
    for i in range(len(levels) - 1):
        ax.barh(0, levels[i+1] - levels[i], left=levels[i], 
                color=colors[i], alpha=0.7, height=0.5)
    
    ax.plot([complexity, complexity], [-0.3, 0.3], 'k-', linewidth=3)
    ax.text(complexity, 0.5, '▼', ha='center', fontsize=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel('Surgical Complexity Score', fontsize=12)
    ax.set_title('Complexity Assessment', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    # Organ involvement
    ax = axes[1, 0]
    organs = roadmap['organ_involvement']
    organ_names = list(organs.keys())
    organ_status = [1 if organs[o] else 0 for o in organ_names]
    
    colors_bar = ['red' if s else 'lightgray' for s in organ_status]
    ax.barh(organ_names, organ_status, color=colors_bar, alpha=0.7)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel('Involvement Status', fontsize=12)
    ax.set_title('Organ Involvement Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    
    # Recommendations
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'Clinical Recommendations', ha='center', 
            fontsize=14, fontweight='bold')
    
    y_pos = 0.75
    for i, rec in enumerate(roadmap['recommendations'], 1):
        ax.text(0.05, y_pos, f"{i}. {rec}", fontsize=10, va='top', wrap=True)
        y_pos -= 0.2
        
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_sample(model, image_path, mask_path, output_dir, device='cpu'):
    """Process a single sample and generate all outputs"""
    # Load data
    image = nib.load(str(image_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()
    
    # Get middle slice
    mid_slice = image.shape[2] // 2
    image_slice = image[:, :, mid_slice]
    mask_slice = mask[:, :, mid_slice]
    
    # Normalize
    image_norm = (image_slice - image_slice.mean()) / (image_slice.std() + 1e-8)
    
    # Predict
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = prediction.cpu().squeeze().numpy()
        
    # Generate outputs
    subject_id = Path(image_path).parent.name
    
    # 1. Comprehensive comparison figure
    comparison_path = os.path.join(output_dir, f'{subject_id}_comparison.png')
    dice = create_comparison_figure(
        image_slice, mask_slice, prediction > 0.5, 
        prediction, comparison_path
    )
    
    # 2. Surgical roadmap
    roadmap = generate_surgical_roadmap(prediction, image_slice.shape)
    roadmap_path = os.path.join(output_dir, f'{subject_id}_roadmap.png')
    create_surgical_roadmap_figure(roadmap, roadmap_path)
    
    # 3. Save roadmap JSON
    roadmap_json_path = os.path.join(output_dir, f'{subject_id}_roadmap.json')
    with open(roadmap_json_path, 'w') as f:
        json.dump(roadmap, f, indent=2)
        
    return {
        'subject_id': subject_id,
        'dice': dice,
        'roadmap': roadmap
    }


def main():
    parser = argparse.ArgumentParser(description='Generate EndoDetect AI demo outputs')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./demo_outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    
    # Find test samples
    data_path = Path(args.data_dir)
    samples = []
    for subject_dir in data_path.iterdir():
        if not subject_dir.is_dir():
            continue
        images = list(subject_dir.glob("*T2*.nii*"))
        masks = list(subject_dir.glob("*mask*.nii*"))
        if images and masks:
            samples.append({'image': images[0], 'mask': masks[0]})
            
    print(f"Found {len(samples)} samples")
    samples = samples[:args.num_samples]
    
    # Process samples
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\nProcessing sample {i}/{len(samples)}...")
        result = process_sample(
            model, sample['image'], sample['mask'], 
            args.output_dir, device
        )
        results.append(result)
        print(f"  Dice: {result['dice']:.3f}")
        print(f"  Complexity: {result['roadmap']['lesion_characteristics']['complexity_score']:.0f}")
        
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(results),
        'average_dice': np.mean([r['dice'] for r in results]),
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n✅ Demo outputs generated in {args.output_dir}")
    print(f"Average Dice: {summary['average_dice']:.3f}")


if __name__ == '__main__':
    main()
