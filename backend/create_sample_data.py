#!/usr/bin/env python3
"""
Generate synthetic sample data matching published dataset characteristics
For demo purposes - real validation will use actual datasets
"""

import numpy as np
import nibabel as nib
import json
import os
from pathlib import Path

def create_synthetic_mri_with_mask(output_dir, patient_id, has_lesion=True):
    """Create synthetic MRI scan with corresponding mask"""
    # Typical pelvic MRI dimensions
    shape = (256, 256, 32)  # Smaller for demo
    
    # Generate synthetic MRI with realistic intensity distribution
    mri = np.random.randn(*shape) * 100 + 500
    mri = np.clip(mri, 0, 1000)
    
    # Add anatomical structure simulation
    center_y, center_x = shape[0]//2, shape[1]//2
    
    # Simulate uterus (bright region)
    for z_slice in range(8, 24):  # Central slices
        y_grid, x_grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        uterus_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) < 2500
        mri[:,:,z_slice][uterus_mask] += 200
    
    # Create mask
    mask = np.zeros(shape, dtype=np.uint8)
    
    if has_lesion:
        # Simulate lesion (ovarian endometrioma)
        lesion_y, lesion_x = center_y + 30, center_x + 40
        for z_slice in range(12, 20):  # Lesion spans fewer slices
            y_grid, x_grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            lesion_mask = ((y_grid - lesion_y)**2 + (x_grid - lesion_x)**2) < 400
            mri[:,:,z_slice][lesion_mask] += 150
            mask[:,:,z_slice][lesion_mask] = 1
    
    # Save as NIfTI
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(exist_ok=True)
    
    # Save MRI
    mri_img = nib.Nifti1Image(mri.astype(np.float32), np.eye(4))
    nib.save(mri_img, patient_dir / f"{patient_id}_T2.nii.gz")
    
    # Save mask
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
    nib.save(mask_img, patient_dir / f"{patient_id}_mask.nii.gz")
    
    return patient_dir


def create_blood_markers_data(output_file):
    """Generate synthetic blood marker data matching NHANES statistics"""
    patients = []
    
    for i in range(20):
        has_endo = i < 10
        
        if has_endo:
            # Elevated inflammatory markers in endometriosis
            crp = np.random.gamma(3, 2)  # mg/L
            esr = np.random.gamma(15, 2)  # mm/hr
            wbc = np.random.normal(8.5, 1.5)  # 10^3/μL
            neutrophils = np.random.normal(65, 8)  # %
        else:
            # Normal range
            crp = np.random.gamma(1.5, 1)
            esr = np.random.gamma(8, 1.5)
            wbc = np.random.normal(7.0, 1.2)
            neutrophils = np.random.normal(60, 7)
        
        patients.append({
            'patient_id': f'P{i+1:03d}',
            'diagnosis': 'endometriosis' if has_endo else 'control',
            'crp_mg_l': round(crp, 2),
            'esr_mm_hr': round(esr, 1),
            'wbc_count': round(wbc, 2),
            'neutrophil_pct': round(neutrophils, 1),
            'nlr': round((neutrophils / (100 - neutrophils)) * wbc, 2),
            'age': np.random.randint(25, 45),
            'bmi': round(np.random.normal(26, 4), 1)
        })
    
    with open(output_file, 'w') as f:
        json.dump(patients, f, indent=2)
    
    return len(patients)


def create_phenotype_data(output_file):
    """Generate WERF EPHect-based phenotype data"""
    phenotypes = []
    
    pain_descriptors = ['sharp', 'dull', 'cramping', 'burning']
    locations = ['lower abdomen', 'pelvic', 'lower back', 'rectum']
    
    for i in range(10):
        phenotypes.append({
            'patient_id': f'P{i+1:03d}',
            'dysmenorrhea_vas': np.random.randint(6, 10),  # 0-10 scale
            'chronic_pelvic_pain_vas': np.random.randint(4, 9),
            'dyspareunia_vas': np.random.randint(3, 8),
            'pain_descriptor': np.random.choice(pain_descriptors),
            'pain_location': np.random.choice(locations),
            'infertility': bool(np.random.choice([True, False], p=[0.4, 0.6])),
            'previous_surgery': bool(np.random.choice([True, False], p=[0.3, 0.7])),
            'werf_stage': np.random.choice(['I', 'II', 'III', 'IV'], p=[0.1, 0.3, 0.4, 0.2]),
            'qol_score': np.random.randint(30, 70)  # SF-36 equivalent
        })
    
    with open(output_file, 'w') as f:
        json.dump(phenotypes, f, indent=2)
    
    return len(phenotypes)


def main():
    base_dir = Path('/Users/azrabano/EndoDetect-AI/data/sample_datasets')
    
    print("Generating synthetic sample datasets...")
    print("(Based on published dataset characteristics)")
    print()
    
    # 1. Create MRI samples (UTHealth-style)
    print("1. Creating MRI samples (UT-EndoMRI style)...")
    mri_dir = base_dir / 'mri_samples'
    mri_dir.mkdir(exist_ok=True)
    
    for i in range(1, 6):
        has_lesion = i <= 3  # 3 with lesions, 2 controls
        patient_id = f'MRI_{i:03d}'
        create_synthetic_mri_with_mask(mri_dir, patient_id, has_lesion)
        status = "endometriosis" if has_lesion else "control"
        print(f"   Created {patient_id} ({status})")
    
    print(f"   ✓ {5} MRI samples created")
    print()
    
    # 2. Create blood marker data (NHANES-style)
    print("2. Creating blood marker data (NHANES style)...")
    blood_file = base_dir / 'blood_markers.json'
    n_blood = create_blood_markers_data(blood_file)
    print(f"   ✓ {n_blood} patient blood marker profiles created")
    print()
    
    # 3. Create phenotype data (WERF EPHect)
    print("3. Creating clinical phenotype data (WERF EPHect)...")
    pheno_file = base_dir / 'clinical_phenotypes.json'
    n_pheno = create_phenotype_data(pheno_file)
    print(f"   ✓ {n_pheno} patient phenotype records created")
    print()
    
    # 4. Create dataset manifest
    print("4. Creating dataset manifest...")
    manifest = {
        'datasets': {
            'mri_samples': {
                'source': 'UT-EndoMRI (synthetic sample)',
                'n_patients': 5,
                'modality': 'T2-weighted MRI',
                'has_annotations': True,
                'location': str(mri_dir.relative_to(base_dir))
            },
            'blood_markers': {
                'source': 'NHANES (synthetic sample)',
                'n_patients': 20,
                'markers': ['CRP', 'ESR', 'CBC', 'NLR'],
                'location': 'blood_markers.json'
            },
            'clinical_phenotypes': {
                'source': 'WERF EPHect (synthetic sample)',
                'n_patients': 10,
                'features': ['pain scores', 'QoL', 'infertility', 'staging'],
                'location': 'clinical_phenotypes.json'
            }
        },
        'note': 'These are synthetic samples for demonstration. Real validation uses actual public datasets.',
        'real_datasets_for_validation': [
            'UT-EndoMRI (51 patients, Zenodo)',
            'AIUM/Balica TVUS dataset',
            'nnU-Net DIE segmentation dataset',
            'NHANES inflammatory markers',
            'UK Biobank reproductive health subset'
        ]
    }
    
    manifest_file = base_dir / 'dataset_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"   ✓ Manifest created")
    print()
    
    print("=" * 60)
    print("✅ SAMPLE DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print(f"Location: {base_dir}")
    print()
    print("Contents:")
    print("  - 5 MRI scans with lesion annotations")
    print("  - 20 blood marker profiles (10 endo, 10 control)")
    print("  - 10 clinical phenotype records")
    print()
    print("These samples enable:")
    print("  ✓ Training proof-of-concept models")
    print("  ✓ Generating demo visualizations")
    print("  ✓ Showing multimodal data integration")
    print()
    print("For actual validation, full datasets will be downloaded.")


if __name__ == '__main__':
    main()
