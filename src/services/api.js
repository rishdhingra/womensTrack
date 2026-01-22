/**
 * EndoDetect AI - API Integration Service
 * Simplified API for radiomics-based inference
 */

const API_BASE_URL = 'http://localhost:5001/api';

/**
 * Upload image file to backend
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || `Upload failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.file_id;
}

/**
 * Run radiomics inference on uploaded file
 */
export async function runInference(fileId) {
  const response = await fetch(`${API_BASE_URL}/inference`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file_id: fileId }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || `Inference failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get saved results by result ID
 */
export async function getResults(resultId) {
  const response = await fetch(`${API_BASE_URL}/results/${resultId}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || `Failed to fetch results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get model info and metadata
 */
export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model-info`);

  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Health check
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}
