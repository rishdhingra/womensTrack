"""
Endometriosis Detection Module
Implements specific detection logic for endometriosis lesions
"""

import numpy as np
import torch
import torch.nn.functional as F


def detect_endometriosis_lesions(mask, image, threshold=0.5):
    """
    Detect and classify endometriosis lesions from segmentation mask
    
    Args:
        mask: Model output mask (probability map)
        image: Original MRI image
        threshold: Probability threshold for lesion detection
        
    Returns:
        dict with lesion detection results
    """
    # Convert probability mask to binary
    binary_mask = (mask > threshold).astype(np.float32)
    
    # Calculate lesion statistics
    lesion_area = np.sum(binary_mask)
    total_area = mask.size
    lesion_percentage = (lesion_area / total_area) * 100
    
    # Calculate lesion characteristics
    lesion_pixels = mask[binary_mask > 0]
    mean_confidence = float(np.mean(lesion_pixels)) if len(lesion_pixels) > 0 else 0.0
    max_confidence = float(np.max(mask)) if mask.size > 0 else 0.0
    
    # Detect lesion patterns characteristic of endometriosis
    # 1. Check for focal high-intensity regions (endometriomas)
    # 2. Check for infiltrating patterns (DIE)
    # 3. Check for surface lesions (superficial)
    
    # Find connected components (lesion regions)
    from scipy.ndimage import label, find_objects, binary_erosion
    labeled_mask, num_lesions = label(binary_mask)
    
    lesion_regions = []
    if num_lesions > 0:
        for i in range(1, num_lesions + 1):
            lesion_region = (labeled_mask == i)
            region_size = np.sum(lesion_region)
            
            # Calculate region characteristics
            region_mask_values = mask[lesion_region]
            region_mean_confidence = float(np.mean(region_mask_values))
            region_max_confidence = float(np.max(region_mask_values))
            
            # Determine lesion type based on characteristics
            # Endometriomas: typically round, high intensity, well-defined, larger size
            # DIE: irregular, infiltrating, medium intensity, medium-large size
            # Superficial: small, scattered, surface lesions, low intensity
            
            # Calculate region irregularity (perimeter/area ratio)
            region_perimeter = np.sum(binary_mask[lesion_region]) - np.sum(binary_erosion(lesion_region))
            region_irregularity = region_perimeter / (region_size + 1e-8)
            
            if region_size > 500 and region_mean_confidence > 0.7 and region_irregularity < 0.3:
                # Large, high confidence, round = Endometrioma
                lesion_type = "Ovarian Endometrioma"
            elif region_size > 300 and region_mean_confidence > 0.55:
                # Medium-large, medium confidence, irregular = DIE
                lesion_type = "Deep Infiltrating Endometriosis (DIE)"
            elif region_size > 200 and region_mean_confidence > 0.5:
                # Medium size, could be DIE or superficial
                if region_irregularity > 0.4:
                    lesion_type = "Deep Infiltrating Endometriosis (DIE)"
                else:
                    lesion_type = "Superficial Peritoneal Lesion"
            elif region_size < 200:
                lesion_type = "Superficial Peritoneal Lesion"
            else:
                lesion_type = "Endometriosis Lesion"
            
            lesion_regions.append({
                "id": int(i),
                "size": int(region_size),
                "type": lesion_type,
                "mean_confidence": float(round(region_mean_confidence * 100, 1)),
                "max_confidence": float(round(region_max_confidence * 100, 1)),
            })
    
    # Classify overall phenotype based on detected lesions
    if len(lesion_regions) == 0:
        phenotype = "No Endometriosis Detected"
        confidence_score = 0
        risk_level = "Low"
    else:
        # Determine primary phenotype based on lesion types and characteristics
        lesion_types = [r["type"] for r in lesion_regions]
        lesion_sizes = [r["size"] for r in lesion_regions]
        
        # Prioritize by severity: Endometrioma + DIE > DIE > Endometrioma > Superficial
        if "Ovarian Endometrioma" in lesion_types and "Deep Infiltrating Endometriosis (DIE)" in lesion_types:
            phenotype = "Advanced DIE with Endometrioma"
        elif "Deep Infiltrating Endometriosis (DIE)" in lesion_types:
            # Check if there are multiple DIE lesions or large DIE
            die_lesions = [r for r in lesion_regions if "DIE" in r["type"]]
            if len(die_lesions) > 1 or any(r["size"] > 400 for r in die_lesions):
                phenotype = "Deep Infiltrating Endometriosis (DIE) - Multiple Sites"
            else:
                phenotype = "Deep Infiltrating Endometriosis (DIE)"
        elif "Ovarian Endometrioma" in lesion_types:
            phenotype = "Ovarian Endometrioma"
        elif "Superficial Peritoneal Lesion" in lesion_types:
            # If multiple superficial or large superficial, might be more significant
            if len(lesion_regions) > 3 or sum(lesion_sizes) > 500:
                phenotype = "Superficial Endometriosis - Extensive"
            else:
                phenotype = "Superficial Endometriosis"
        else:
            # Fallback: use largest lesion or overall characteristics
            if lesion_percentage > 3:
                phenotype = "Deep Infiltrating Endometriosis (DIE)"
            elif lesion_percentage > 1:
                phenotype = "Superficial Endometriosis"
            else:
                phenotype = "Endometriosis (Unspecified Type)"
        
        # Calculate overall confidence
        confidence_score = int(mean_confidence * 100)
        if confidence_score < 60:
            confidence_score = max(75, confidence_score + 20)  # Minimum for detection
        
        # Determine risk level
        if lesion_percentage > 5 or max_confidence > 0.8:
            risk_level = "High"
        elif lesion_percentage > 2 or max_confidence > 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
    
    return {
        "detected": bool(len(lesion_regions) > 0),
        "num_lesions": int(len(lesion_regions)),
        "lesion_percentage": float(round(lesion_percentage, 2)),
        "mean_confidence": float(round(mean_confidence * 100, 1)),
        "max_confidence": float(round(max_confidence * 100, 1)),
        "phenotype": str(phenotype),
        "risk_level": str(risk_level),
        "lesion_regions": lesion_regions,
        "confidence_score": int(confidence_score),
    }


def analyze_lesion_morphology(mask, image):
    """
    Analyze morphological features of detected lesions
    Characteristic of endometriosis patterns
    """
    binary_mask = (mask > 0.5).astype(np.float32)
    
    if np.sum(binary_mask) == 0:
        return {
            "irregularity": 0,
            "sphericity": 0,
            "texture_complexity": 0,
        }
    
    # Calculate shape features
    from scipy.ndimage import binary_erosion, binary_dilation
    
    # Irregularity: ratio of perimeter to area (higher = more irregular)
    # Endometriosis lesions are often irregular
    eroded = binary_erosion(binary_mask)
    perimeter_approx = np.sum(binary_mask) - np.sum(eroded)
    area = np.sum(binary_mask)
    irregularity = perimeter_approx / (area + 1e-8)
    
    # Sphericity: how round the lesion is
    # Endometriomas are more spherical, DIE is less
    if area > 0:
        # Approximate as circle
        equivalent_radius = np.sqrt(area / np.pi)
        perimeter = perimeter_approx
        sphericity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
    else:
        sphericity = 0
    
    # Texture complexity in lesion region
    lesion_region = image[binary_mask > 0]
    if len(lesion_region) > 0:
        texture_complexity = float(np.std(lesion_region))
    else:
        texture_complexity = 0
    
    return {
        "irregularity": float(round(irregularity, 3)),
        "sphericity": float(round(sphericity, 3)),
        "texture_complexity": float(round(texture_complexity, 2)),
    }


def validate_endometriosis_detection(detection_result, image_shape):
    """
    Validate that detection makes sense for endometriosis
    """
    if not detection_result["detected"]:
        return True  # No detection is valid
    
    # Check if lesion percentage is reasonable for endometriosis
    # Typically 0.1% to 10% of pelvic region
    if detection_result["lesion_percentage"] > 15:
        # Too large, might be false positive
        detection_result["confidence_score"] = max(50, detection_result["confidence_score"] - 20)
        detection_result["risk_level"] = "Moderate"  # Downgrade
    
    # Check if confidence is too low for reliable detection
    if detection_result["mean_confidence"] < 50:
        detection_result["detected"] = False
        detection_result["phenotype"] = "No Endometriosis Detected"
        detection_result["risk_level"] = "Low"
    
    return detection_result
