/**
 * Sample Patient Data for EndoDetect AI Demo
 * 
 * Four cohorts demonstrating D³ framework:
 * 1. DIE with High Pain - Shows Diagnostics + Drivers (inflammatory endotype)
 * 2. Ovarian Endometrioma with Minimal Pain - Shows phenotype heterogeneity
 * 3. Chronic Pelvic Pain without visible endo - Shows Drivers (neuroimmune)
 * 4. Gynecologic Control - Negative case for comparison
 */

export const samplePatients = {
  // CASE 1: Deep Infiltrating Endometriosis (DIE) - High Pain, Inflammatory Endotype
  die_high_pain: {
    id: "PT-001",
    name: "Patient A - DIE with High Pain",
    cohort: "Endometriosis + High Pain",
    description: "32-year-old with severe dysmenorrhea, dyschezia, and chronic pelvic pain. Demonstrates inflammatory endotype.",
    
    // Demographics & Clinical
    demographics: {
      age: 32,
      bmi: 24.3,
      nulliparous: true,
      symptom_duration_years: 5,
    },
    
    // WERF EPHect Phenotyping
    phenotype: {
      pain_scores: {
        dysmenorrhea: 9,
        dyspareunia: 7,
        dyschezia: 8,
        dysuria: 4,
        chronic_pelvic_pain: 8,
      },
      menstrual_flow: "heavy",
      infertility: true,
    },
    
    // Imaging Results (Aim 1 - Diagnostics)
    imaging: {
      modality: "MRI T2W + TVUS",
      acquisition_date: "2025-01-10",
      phenotype_classification: {
        die: { probability: 0.89, confidence: "high" },
        ovarian: { probability: 0.12, confidence: "low" },
        superficial: { probability: 0.34, confidence: "medium" },
      },
      lesion_locations: [
        { organ: "Rectosigmoid", involvement: 85, depth_mm: 12 },
        { organ: "Uterosacral Ligaments", involvement: 92, depth_mm: 8 },
        { organ: "Posterior Cul-de-sac", involvement: 78, depth_mm: 15 },
        { organ: "Bladder", involvement: 23, depth_mm: 3 },
      ],
      dice_score: 0.82,
    },
    
    // Radiomics Features (100+ features, showing key ones)
    radiomics: {
      first_order: {
        mean_intensity: 342.5,
        std_intensity: 87.3,
        skewness: 0.45,
        kurtosis: 2.87,
        entropy: 5.23,
      },
      shape: {
        volume_mm3: 2847,
        surface_area_mm2: 1523,
        sphericity: 0.34,
        compactness: 0.41,
      },
      texture_glcm: {
        contrast: 127.4,
        correlation: 0.72,
        energy: 0.15,
        homogeneity: 0.58,
      },
      texture_glrlm: {
        run_entropy: 4.82,
        run_percentage: 0.45,
        long_run_emphasis: 2.34,
      },
      wavelet: {
        hh_mean: 23.4,
        hl_mean: 45.7,
        lh_mean: 38.2,
        ll_mean: 189.3,
      },
    },
    
    // Blood Biomarkers (Aim 2 - Drivers: Inflammatory Endotype)
    blood_biomarkers: {
      wbc_count: 9200,
      neutrophils: 6440, // 70%
      lymphocytes: 2024, // 22%
      nlr: 3.18, // Elevated - inflammatory signature
      crp_mg_L: 8.4, // Elevated
      esr_mm_hr: 24, // Elevated
      inflammatory_endotype: "High Inflammatory",
      correlation_strength: 0.76,
    },
    
    // Surgical Roadmap (Clinical Use)
    surgical_roadmap: {
      complexity: "High",
      estimated_or_time_min: 210,
      recommended_team: [
        "Minimally Invasive Gynecologic Surgeon",
        "Colorectal Surgeon (for bowel resection)",
        "Urologist (standby)",
      ],
      surgical_approach: "Laparoscopic with possible bowel resection",
      preop_recommendations: [
        "Bowel prep required",
        "Ureteral stent placement consideration",
        "MIS + colorectal co-surgery",
      ],
      recurrence_risk: "High (65%)",
    },
    
    // Biobank Priority (Aim 3 - Development)
    biobank: {
      priority: "High",
      rationale: "Inflammatory endotype + DIE + high pain - ideal for biomarker discovery",
      samples_to_collect: [
        "Lesion tissue (rectosigmoid nodule)",
        "Eutopic endometrium",
        "Peritoneal fluid",
        "Serum (inflammatory panel)",
        "Urine",
      ],
    },
  },

  // CASE 2: Ovarian Endometrioma - Minimal Pain
  ovarian_minimal_pain: {
    id: "PT-002",
    name: "Patient B - Ovarian Endometrioma",
    cohort: "Endometriosis + Minimal Pain",
    description: "28-year-old with bilateral ovarian cysts, minimal symptoms. Demonstrates low inflammatory endotype.",
    
    demographics: {
      age: 28,
      bmi: 22.1,
      nulliparous: true,
      symptom_duration_years: 2,
    },
    
    phenotype: {
      pain_scores: {
        dysmenorrhea: 3,
        dyspareunia: 2,
        dyschezia: 1,
        dysuria: 0,
        chronic_pelvic_pain: 2,
      },
      menstrual_flow: "moderate",
      infertility: false,
    },
    
    imaging: {
      modality: "TVUS + MRI",
      acquisition_date: "2025-01-08",
      phenotype_classification: {
        die: { probability: 0.15, confidence: "low" },
        ovarian: { probability: 0.94, confidence: "very_high" },
        superficial: { probability: 0.28, confidence: "low" },
      },
      lesion_locations: [
        { organ: "Left Ovary", involvement: 78, diameter_mm: 42 },
        { organ: "Right Ovary", involvement: 65, diameter_mm: 35 },
        { organ: "Anterior Cul-de-sac", involvement: 15, depth_mm: 2 },
      ],
      dice_score: 0.87,
    },
    
    radiomics: {
      first_order: {
        mean_intensity: 285.2,
        std_intensity: 62.1,
        skewness: -0.23,
        kurtosis: 1.98,
        entropy: 4.56,
      },
      shape: {
        volume_mm3: 18542,
        surface_area_mm2: 4523,
        sphericity: 0.78, // More spherical (classic endometrioma)
        compactness: 0.71,
      },
      texture_glcm: {
        contrast: 45.2,
        correlation: 0.85,
        energy: 0.28,
        homogeneity: 0.74,
      },
      texture_glrlm: {
        run_entropy: 3.92,
        run_percentage: 0.62,
        long_run_emphasis: 1.67,
      },
      wavelet: {
        hh_mean: 18.7,
        hl_mean: 32.4,
        lh_mean: 28.9,
        ll_mean: 156.8,
      },
    },
    
    blood_biomarkers: {
      wbc_count: 6800,
      neutrophils: 3808, // 56%
      lymphocytes: 2380, // 35%
      nlr: 1.60, // Normal - low inflammatory
      crp_mg_L: 1.2, // Normal
      esr_mm_hr: 8, // Normal
      inflammatory_endotype: "Low Inflammatory",
      correlation_strength: 0.42,
    },
    
    surgical_roadmap: {
      complexity: "Moderate",
      estimated_or_time_min: 90,
      recommended_team: [
        "Minimally Invasive Gynecologic Surgeon",
      ],
      surgical_approach: "Laparoscopic cystectomy",
      preop_recommendations: [
        "Standard prep",
        "Preserve ovarian reserve",
        "Consider fertility counseling",
      ],
      recurrence_risk: "Moderate (35%)",
    },
    
    biobank: {
      priority: "Medium",
      rationale: "Low pain + ovarian phenotype - useful for pain mechanism comparison",
      samples_to_collect: [
        "Ovarian cyst wall",
        "Eutopic endometrium",
        "Serum",
        "Urine",
      ],
    },
  },

  // CASE 3: Chronic Pelvic Pain WITHOUT Visible Endometriosis
  cpp_no_endo: {
    id: "PT-003",
    name: "Patient C - CPP without Visible Endo",
    cohort: "Chronic Pelvic Pain (No Visible Endo)",
    description: "35-year-old with severe pain but no visible lesions on imaging. May have microscopic/neuroinflammatory disease.",
    
    demographics: {
      age: 35,
      bmi: 26.8,
      nulliparous: false,
      symptom_duration_years: 8,
    },
    
    phenotype: {
      pain_scores: {
        dysmenorrhea: 8,
        dyspareunia: 9,
        dyschezia: 6,
        dysuria: 5,
        chronic_pelvic_pain: 9,
      },
      menstrual_flow: "moderate",
      infertility: false,
    },
    
    imaging: {
      modality: "MRI + TVUS",
      acquisition_date: "2025-01-12",
      phenotype_classification: {
        die: { probability: 0.08, confidence: "very_low" },
        ovarian: { probability: 0.05, confidence: "very_low" },
        superficial: { probability: 0.18, confidence: "low" },
      },
      lesion_locations: [],
      dice_score: null, // No visible lesions
      notes: "No macroscopic lesions detected. Consider microscopic disease or central sensitization.",
    },
    
    radiomics: {
      first_order: {
        mean_intensity: 298.7,
        std_intensity: 71.2,
        skewness: 0.12,
        kurtosis: 2.34,
        entropy: 4.89,
      },
      shape: null, // No discrete lesions
      texture_glcm: {
        contrast: 78.3,
        correlation: 0.68,
        energy: 0.19,
        homogeneity: 0.62,
      },
      texture_glrlm: {
        run_entropy: 4.45,
        run_percentage: 0.51,
        long_run_emphasis: 1.98,
      },
      wavelet: {
        hh_mean: 21.2,
        hl_mean: 39.8,
        lh_mean: 34.1,
        ll_mean: 174.5,
      },
    },
    
    // Key finding: Elevated inflammatory markers WITHOUT visible lesions
    blood_biomarkers: {
      wbc_count: 8900,
      neutrophils: 6230, // 70%
      lymphocytes: 1780, // 20%
      nlr: 3.50, // Elevated - suggests systemic inflammation
      crp_mg_L: 6.8, // Elevated
      esr_mm_hr: 21, // Elevated
      inflammatory_endotype: "Neuroimmune/Central Sensitization",
      correlation_strength: 0.68,
      notes: "High inflammatory markers without visible lesions - suggests microscopic or neuroimmune pathology",
    },
    
    surgical_roadmap: {
      complexity: "Low (Diagnostic)",
      estimated_or_time_min: 45,
      recommended_team: [
        "Gynecologic Surgeon",
      ],
      surgical_approach: "Diagnostic laparoscopy with peritoneal biopsies",
      preop_recommendations: [
        "Consider pain management referral",
        "Peritoneal biopsies for microscopic endo",
        "Document pain mapping during surgery",
      ],
      recurrence_risk: "Unknown (no visible disease)",
    },
    
    biobank: {
      priority: "Very High",
      rationale: "Critical case for understanding pain without visible disease - neuroimmune mechanisms",
      samples_to_collect: [
        "Peritoneal biopsies (multiple sites)",
        "Eutopic endometrium",
        "Peritoneal fluid",
        "Serum (inflammatory + neuroimmune panels)",
        "Urine",
      ],
    },
  },

  // CASE 4: Gynecologic Control (Negative Case)
  control: {
    id: "PT-004",
    name: "Patient D - Control",
    cohort: "Gynecologic Control",
    description: "30-year-old undergoing surgery for benign indication (fibroid), no endometriosis.",
    
    demographics: {
      age: 30,
      bmi: 23.5,
      nulliparous: false,
      symptom_duration_years: 0,
    },
    
    phenotype: {
      pain_scores: {
        dysmenorrhea: 1,
        dyspareunia: 0,
        dyschezia: 0,
        dysuria: 0,
        chronic_pelvic_pain: 0,
      },
      menstrual_flow: "moderate",
      infertility: false,
    },
    
    imaging: {
      modality: "MRI",
      acquisition_date: "2025-01-14",
      phenotype_classification: {
        die: { probability: 0.02, confidence: "very_low" },
        ovarian: { probability: 0.03, confidence: "very_low" },
        superficial: { probability: 0.05, confidence: "very_low" },
      },
      lesion_locations: [],
      dice_score: null,
      notes: "No endometriosis detected. Uterine fibroid present (not analyzed).",
    },
    
    radiomics: {
      first_order: {
        mean_intensity: 312.4,
        std_intensity: 58.9,
        skewness: 0.05,
        kurtosis: 2.12,
        entropy: 4.23,
      },
      shape: null,
      texture_glcm: {
        contrast: 52.1,
        correlation: 0.81,
        energy: 0.31,
        homogeneity: 0.76,
      },
      texture_glrlm: {
        run_entropy: 3.67,
        run_percentage: 0.68,
        long_run_emphasis: 1.45,
      },
      wavelet: {
        hh_mean: 16.8,
        hl_mean: 28.9,
        lh_mean: 25.3,
        ll_mean: 142.7,
      },
    },
    
    blood_biomarkers: {
      wbc_count: 7100,
      neutrophils: 4260, // 60%
      lymphocytes: 2485, // 35%
      nlr: 1.71, // Normal
      crp_mg_L: 0.8, // Normal
      esr_mm_hr: 6, // Normal
      inflammatory_endotype: "Normal",
      correlation_strength: null,
    },
    
    surgical_roadmap: {
      complexity: "Low",
      estimated_or_time_min: 60,
      recommended_team: [
        "Gynecologic Surgeon",
      ],
      surgical_approach: "Laparoscopic myomectomy",
      preop_recommendations: [
        "Standard prep for fibroid removal",
      ],
      recurrence_risk: "N/A",
    },
    
    biobank: {
      priority: "Medium",
      rationale: "Control sample - essential for comparative studies",
      samples_to_collect: [
        "Eutopic endometrium",
        "Serum",
        "Urine",
      ],
    },
  },
};

// Export individual cases for easy access
export const dieHighPain = samplePatients.die_high_pain;
export const ovarianMinimalPain = samplePatients.ovarian_minimal_pain;
export const cppNoEndo = samplePatients.cpp_no_endo;
export const control = samplePatients.control;

// Export list of all patient IDs and names for UI selector
export const patientList = [
  { id: "die_high_pain", label: "DIE + High Pain (Inflammatory)", tag: "Aim 1 + 2" },
  { id: "ovarian_minimal_pain", label: "Ovarian Endo + Low Pain", tag: "Aim 1" },
  { id: "cpp_no_endo", label: "CPP without Visible Endo", tag: "Aim 2" },
  { id: "control", label: "Control (No Endometriosis)", tag: "Aim 3" },
];
