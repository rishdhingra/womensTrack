/**
 * EndoDetect AI - API Integration Service
 * Handles all communication with Flask backend
 */

const API_BASE_URL = 'http://localhost:5000/api';

class ApiService {
  /**
   * Upload image file to backend
   */
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Run model inference on uploaded file
   */
  async runInference(fileId) {
    const response = await fetch(`${API_BASE_URL}/inference`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_id: fileId }),
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get radiomics features (100+) for uploaded file
   */
  async getRadiomicsFeatures(fileId) {
    const response = await fetch(`${API_BASE_URL}/radiomics-features/${fileId}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch radiomics: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Correlate imaging features with blood biomarkers
   */
  async getBloodCorrelation(imagingFeatures, bloodMarkers) {
    const response = await fetch(`${API_BASE_URL}/blood-correlation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imaging_features: imagingFeatures,
        blood_markers: bloodMarkers,
      }),
    });

    if (!response.ok) {
      throw new Error(`Blood correlation failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get patient stratification for Aim 3 cohorts
   */
  async getPatientStratification(patientData) {
    const response = await fetch(`${API_BASE_URL}/patient-stratification`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patientData),
    });

    if (!response.ok) {
      throw new Error(`Stratification failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get model info and metadata
   */
  async getModelInfo() {
    const response = await fetch(`${API_BASE_URL}/model-info`);

    if (!response.ok) {
      throw new Error(`Failed to fetch model info: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Health check
   */
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Full workflow: Upload + Inference
   */
  async processImage(file) {
    try {
      // 1. Upload file
      const uploadResult = await this.uploadFile(file);
      const fileId = uploadResult.file_id;

      // 2. Run inference
      const inferenceResult = await this.runInference(fileId);

      // 3. Get radiomics features
      const radiomicsResult = await this.getRadiomicsFeatures(fileId);

      return {
        fileId,
        inference: inferenceResult,
        radiomics: radiomicsResult,
      };
    } catch (error) {
      console.error('Error processing image:', error);
      throw error;
    }
  }
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;
