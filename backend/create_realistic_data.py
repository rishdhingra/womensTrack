#!/usr/bin/env python3
"""
Generate realistic synthetic endometriosis MRI data
Modeled after UT-EndoMRI dataset specifications (Liang et al. 2025)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom
import json

def create_realistic_anatomy(shape=(256, 256, 48)):
    """Create realistic pelvic anatomy with proper intensity distributions"""
    image = np.zeros(shape)
    
    # Background (air/fat): low intensity
    image += np.random.randn(*shape) * 20 + 100
    
    # Bladder (anterior, bright on T2): high signal
    y, x, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    bladder = ((y - 90)**2/1200 + (x - 128)**2/1500 + (z - 24)**2/400) < 1
    image[bladder] = np.random.randn(np.sum(bladder)) * 30 + 800
    
    # Uterus (central): intermediate-high signal
    uterus = ((y - 140)**2/800 + (x - 128)**2/900 + (z - 24)**2/300) < 1
    image[uterus] = np.random.randn(np.sum(uterus)) * 40 + 600
    
    # Ovaries (bilateral, posterior to uterus): intermediate signal
    # Left ovary
    left_ovary = ((y - 150)**2/200 + (x - 90)**2/180 + (z - 24)**2/150) < 1
    image[left_ovary] = np.random.randn(np.sum(left_ovary)) * 35 + 500
    
    # Right ovary
    right_ovary = ((y - 150)**2/200 + (x - 166)**2/180 + (z - 24)**2/150) < 1
    image[right_ovary] = np.random.randn(np.sum(right_ovary)) * 35 + 500
    
    # Rectum (posterior): variable signal
    rectum = ((y - 180)**2/600 + (x - 128)**2/700 + (z - 24)**2/350) < 1
    image[rectum] = np.random.randn(np.sum(rectum)) * 25 + 250
    
    # Apply Gaussian smoothing for realistic appearance
    image = gaussian_filter(image, sigma=1.5)
    
    # Add slice-to-slice variation
    for z_slice in range(shape[2]):
        intensity_var = np.random.uniform(0.95, 1.05)
        image[:, :, z_slice] *= intensity_var
    
    return image


def create_endometriosis_lesions(image_shape, lesion_type='DIE', severity='moderate'):
    """Create realistic endometriosis lesion masks"""
    mask = np.zeros(image_shape)
    y, x, z = np.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
    
    if lesion_type == 'ovarian' or lesion_type == 'mixed':
        # Ovarian endometrioma (chocolate cyst) - left ovary
        if severity in ['moderate', 'severe']:
            endometrioma = ((y - 150)**2/120 + (x - 90)**2/100 + (z - 24)**2/80) < 1
            mask[endometrioma] = 2  # Class 2: endometrioma
    
    if lesion_type == 'DIE' or lesion_type == 'mixed':
        # Deep infiltrating endometriosis - uterosacral ligaments
        # Left USL
        left_usl = ((y - 165)**2/80 + (x - 105)**2/60 + (z - 22)**2/100) < 1
        mask[left_usl] = 3  # Class 3: DIE
        
        # Right USL
        if severity == 'severe':
            right_usl = ((y - 165)**2/80 + (x - 151)**2/60 + (z - 22)**2/100) < 1
            mask[right_usl] = 3
        
        # Rectovaginal septum involvement (severe cases)
        if severity == 'severe':
            rvs = ((y - 175)**2/100 + (x - 128)**2/90 + (z - 24)**2/70) < 1
            mask[rvs] = 3
    
    if lesion_type == 'superficial' or (lesion_type == 'mixed' and severity == 'severe'):
        # Superficial peritoneal lesions
        # Pouch of Douglas
        pod = ((y - 170)**2/150 + (x - 128)**2/140 + (z - 26)**2/50) < 1
        pod_mask = pod & (np.random.rand(*image_shape) > 0.7)  # Patchy distribution
        mask[pod_mask] = 1  # Class 1: superficial
    
    # Apply slight morphological smoothing
    from scipy.ndimage import binary_dilation
    for class_id in [1, 2, 3]:
        class_mask = mask == class_id
        if np.any(class_mask):
            dilated = binary_dilation(class_mask, iterations=1)
            mask[dilated & (mask == 0)] = class_id
    
    return mask.astype(np.uint8)


def add_lesion_signal_to_image(image, mask):
    """Modify image intensity in lesion regions"""
    # Endometriomas: very high T2 signal (chocolate cyst)
    endometrioma_mask = mask == 2
    image[endometrioma_mask] = np.random.randn(np.sum(endometrioma_mask)) * 50 + 850
    
    # DIE: low-intermediate T2 signal (fibrotic)
    die_mask = mask == 3
    image[die_mask] = np.random.randn(np.sum(die_mask)) * 30 + 400
    
    # Superficial: slightly elevated T2 signal
    superficial_mask = mask == 1
    image[superficial_mask] = np.random.randn(np.sum(superficial_mask)) * 35 + 550
    
    return image


def generate_patient_data(patient_id, has_endo=True, lesion_type='DIE', severity='moderate'):
    """Generate complete patient dataset"""
    shape = (256, 256, 48)
    
    # Create anatomical structure
    image = create_realistic_anatomy(shape)
    
    # Create lesion mask
    if has_endo:
        mask = create_endometriosis_lesions(shape, lesion_type, severity)
        image = add_lesion_signal_to_image(image, mask)
    else:
        mask = np.zeros(shape, dtype=np.uint8)
    
    # Add realistic noise
    noise = np.random.randn(*shape) * 15
    image = image + noise
    image = np.clip(image, 0, 1000)
    
    return image.astype(np.float32), mask


def main():
    output_dir = Path('/Users/azrabano/EndoDetect-AI/data/realistic_mri')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("Creating Realistic Endometriosis MRI Dataset")
    print("Modeled after UT-EndoMRI (Liang et al. 2025)")
    print("=" * 70)
    
    # Patient cohort design (mimicking real clinical distribution)
    patients = [
        # Endometriosis patients (30 patients)
        *[{'id': f'ENDO_DIE_{i:02d}', 'endo': True, 'type': 'DIE', 'severity': 'moderate'} for i in range(1, 11)],
        *[{'id': f'ENDO_DIE_SEV_{i:02d}', 'endo': True, 'type': 'DIE', 'severity': 'severe'} for i in range(1, 6)],
        *[{'id': f'ENDO_OVAR_{i:02d}', 'endo': True, 'type': 'ovarian', 'severity': 'moderate'} for i in range(1, 8)],
        *[{'id': f'ENDO_MIX_{i:02d}', 'endo': True, 'type': 'mixed', 'severity': 'severe'} for i in range(1, 7)],
        # Controls (15 patients)
        *[{'id': f'CONTROL_{i:02d}', 'endo': False, 'type': None, 'severity': None} for i in range(1, 16)],
    ]
    
    manifest = {'patients': [], 'statistics': {}}
    
    for idx, patient in enumerate(patients, 1):
        patient_dir = output_dir / patient['id']
        patient_dir.mkdir(exist_ok=True)
        
        # Generate data
        image, mask = generate_patient_data(
            patient['id'],
            has_endo=patient['endo'],
            lesion_type=patient.get('type'),
            severity=patient.get('severity')
        )
        
        # Save as NIfTI
        affine = np.eye(4)
        affine[0, 0] = 0.7  # 0.7mm x 0.7mm x 3mm (typical MRI resolution)
        affine[1, 1] = 0.7
        affine[2, 2] = 3.0
        
        img_nii = nib.Nifti1Image(image, affine)
        mask_nii = nib.Nifti1Image(mask, affine)
        
        nib.save(img_nii, patient_dir / f"{patient['id']}_T2W.nii.gz")
        nib.save(mask_nii, patient_dir / f"{patient['id']}_mask.nii.gz")
        
        # Calculate statistics
        lesion_volume = np.sum(mask > 0) * 0.7 * 0.7 * 3.0 / 1000  # in mL
        
        manifest['patients'].append({
            'patient_id': patient['id'],
            'diagnosis': 'endometriosis' if patient['endo'] else 'control',
            'lesion_type': patient.get('type'),
            'severity': patient.get('severity'),
            'lesion_volume_ml': float(lesion_volume) if patient['endo'] else 0.0
        })
        
        print(f"  [{idx}/{len(patients)}] Created {patient['id']} - "
              f"{'Endo: ' + str(patient.get('type')) if patient['endo'] else 'Control'}")
    
    # Save manifest
    manifest['statistics'] = {
        'total_patients': len(patients),
        'endometriosis_cases': sum(1 for p in patients if p['endo']),
        'controls': sum(1 for p in patients if not p['endo']),
        'image_dimensions': '256x256x48',
        'voxel_spacing_mm': '0.7x0.7x3.0',
        'modality': 'T2-weighted MRI',
        'reference': 'Modeled after UT-EndoMRI (Liang et al. 2025)',
        'lesion_classes': {
            '0': 'background',
            '1': 'superficial peritoneal',
            '2': 'ovarian endometrioma',
            '3': 'deep infiltrating (DIE)'
        }
    }
    
    with open(output_dir / 'dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Dataset Creation Complete!")
    print("=" * 70)
    print(f"📁 Location: {output_dir}")
    print(f"📊 Total patients: {manifest['statistics']['total_patients']}")
    print(f"   - Endometriosis: {manifest['statistics']['endometriosis_cases']}")
    print(f"   - Controls: {manifest['statistics']['controls']}")
    print(f"🔬 Lesion types: DIE, Ovarian, Superficial, Mixed")
    print(f"💾 Format: NIfTI (.nii.gz)")
    print(f"📐 Resolution: 256x256x48 (0.7x0.7x3.0 mm)")
    print("=" * 70)


if __name__ == '__main__':
    main()
