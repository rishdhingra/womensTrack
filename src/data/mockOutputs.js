export const mockOutputs = {
  phenotype: 'Deep Infiltrating Endometriosis (DIE)',
  confidence: 87,
  risk: 'Moderate',
  nextSteps: [
    'Consider referral to endometriosis specialist for comprehensive evaluation',
    'Multimodal imaging review recommended (MRI + TVUS correlation)',
    'Surgical planning consultation if symptoms persist or worsen',
  ],
};

export const mockChartData = {
  features: [
    { name: 'Texture', value: 0.85 },
    { name: 'Shape', value: 0.72 },
    { name: 'Intensity', value: 0.68 },
    { name: 'Gradient', value: 0.61 },
    { name: 'Wavelet', value: 0.54 },
  ],
  cohort: [
    { name: 'Q1', Current: 0.65, Reference: 0.58 },
    { name: 'Q2', Current: 0.72, Reference: 0.61 },
    { name: 'Q3', Current: 0.78, Reference: 0.64 },
    { name: 'Q4', Current: 0.82, Reference: 0.67 },
  ],
};

