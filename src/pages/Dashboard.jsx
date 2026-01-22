import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Upload, FileText, CheckCircle, XCircle, Loader } from 'lucide-react';
import Sidebar from '../components/Sidebar';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import UploadBox from '../components/UploadBox';
import { uploadFile, runInference, getResults } from '../services/api';

export default function Dashboard() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tab = searchParams.get('tab') || 'upload'; // Default to upload
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFileId, setUploadedFileId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    // Load model info on mount
    fetch('http://localhost:5001/api/model-info')
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => console.error('Failed to load model info:', err));
  }, []);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);
    setUploadedFileId(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadedFileId(null);
    setResults(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const fileId = await uploadFile(selectedFile);
      setUploadedFileId(fileId);
      setError(null);
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleInference = async () => {
    if (!uploadedFileId) {
      setError('Please upload a file first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const inferenceResults = await runInference(uploadedFileId);
      setResults(inferenceResults);
      setError(null);
      // Switch to outputs tab
      navigate('/dashboard?tab=outputs');
    } catch (err) {
      setError(err.message || 'Inference failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const renderUpload = () => (
    <div className="space-y-6">
      <Card>
        <div className="mb-6">
          <h2 className="text-2xl font-serif font-bold text-gray-900 mb-2">
            Upload Medical Image
          </h2>
          <p className="text-gray-600">
            Upload MRI, CT, or ultrasound images for radiomics-based endometriosis detection
          </p>
        </div>

        {modelInfo && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Model:</strong> {modelInfo.model_name || 'Radiomics Classifier'} 
              {' '}• <strong>Accuracy:</strong> {modelInfo.best_accuracy_percentage?.toFixed(1) || 'N/A'}%
              {' '}• <strong>Dataset:</strong> {modelInfo.dataset || 'N/A'}
            </p>
          </div>
        )}

        <div className="space-y-4">
          <UploadBox
            label="Select Image File"
            accept="image/*,.nii,.nii.gz,.dcm"
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onRemove={handleRemoveFile}
          />

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div className="flex gap-3">
            <Button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className="flex items-center gap-2"
            >
              {isUploading ? (
                <>
                  <Loader className="animate-spin" size={18} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={18} />
                  Upload File
                </>
              )}
            </Button>

            {uploadedFileId && (
              <Button
                onClick={handleInference}
                disabled={isProcessing}
                variant="primary"
                className="flex items-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <Loader className="animate-spin" size={18} />
                    Processing...
                  </>
                ) : (
                  <>
                    <FileText size={18} />
                    Run Analysis
                  </>
                )}
              </Button>
            )}
          </div>

          {uploadedFileId && !isProcessing && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800 flex items-center gap-2">
                <CheckCircle size={16} />
                File uploaded successfully. Click "Run Analysis" to process.
              </p>
            </div>
          )}
        </div>
      </Card>

      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Supported File Formats</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 border border-gray-200 rounded-lg">
            <p className="text-sm font-semibold text-gray-900">NIfTI</p>
            <p className="text-xs text-gray-600">.nii, .nii.gz</p>
          </div>
          <div className="text-center p-3 border border-gray-200 rounded-lg">
            <p className="text-sm font-semibold text-gray-900">DICOM</p>
            <p className="text-xs text-gray-600">.dcm</p>
          </div>
          <div className="text-center p-3 border border-gray-200 rounded-lg">
            <p className="text-sm font-semibold text-gray-900">Images</p>
            <p className="text-xs text-gray-600">.png, .jpg</p>
          </div>
          <div className="text-center p-3 border border-gray-200 rounded-lg">
            <p className="text-sm font-semibold text-gray-900">Archive</p>
            <p className="text-xs text-gray-600">.zip</p>
          </div>
        </div>
      </Card>
    </div>
  );

  const renderOutputs = () => {
    if (!results) {
      return (
        <Card>
          <div className="text-center py-12">
            <FileText size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Results Yet</h3>
            <p className="text-gray-600 mb-4">
              Upload an image and run analysis to see results here
            </p>
            <Button onClick={() => navigate('/dashboard?tab=upload')}>
              Go to Upload
            </Button>
          </div>
        </Card>
      );
    }

    const prediction = results.prediction || {};
    const hasEndo = prediction.has_endometriosis;
    const probability = prediction.probability || 0;
    const confidence = prediction.confidence || 0;

    return (
      <div className="space-y-6">
        <Card>
          <div className="mb-6">
            <h2 className="text-2xl font-serif font-bold text-gray-900 mb-2">
              Analysis Results
            </h2>
            <p className="text-gray-600">
              Radiomics-based endometriosis detection results
            </p>
          </div>

          {/* Prediction Result */}
          <div className="mb-6 p-6 rounded-lg border-2" style={{
            borderColor: hasEndo ? '#dc2626' : '#16a34a',
            backgroundColor: hasEndo ? '#fef2f2' : '#f0fdf4'
          }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                {hasEndo ? (
                  <XCircle size={32} className="text-red-600" />
                ) : (
                  <CheckCircle size={32} className="text-green-600" />
                )}
                <div>
                  <h3 className="text-xl font-bold text-gray-900">
                    {hasEndo ? 'Endometriosis Detected' : 'No Endometriosis Detected'}
                  </h3>
                  <p className="text-sm text-gray-600">
                    Result ID: {results.result_id}
                  </p>
                </div>
              </div>
              <Badge variant={hasEndo ? 'danger' : 'success'} className="text-lg px-4 py-2">
                {hasEndo ? 'Positive' : 'Negative'}
              </Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Probability</p>
                <p className="text-2xl font-bold text-gray-900">
                  {(probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Confidence</p>
                <p className="text-2xl font-bold text-gray-900">
                  {(confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Model Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">
                  {results.model_info?.accuracy?.toFixed(1) || 'N/A'}%
                </p>
              </div>
            </div>
          </div>

          {/* Radiomics Features */}
          {results.radiomics_features && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Extracted Radiomics Features ({results.radiomics_features.count})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
                {Object.entries(results.radiomics_features.feature_values || {}).map(([name, value]) => (
                  <div key={name} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <p className="text-xs font-semibold text-gray-700 mb-1">{name}</p>
                    <p className="text-sm text-gray-900">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Information */}
          {results.model_info && (
            <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
              <h4 className="text-sm font-semibold text-gray-900 mb-2">Model Information</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-600">Type:</span>{' '}
                  <span className="font-semibold">{results.model_info.type}</span>
                </div>
                <div>
                  <span className="text-gray-600">Dataset:</span>{' '}
                  <span className="font-semibold">{results.model_info.dataset}</span>
                </div>
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <Button onClick={() => navigate('/dashboard?tab=upload')}>
              Analyze Another Image
            </Button>
          </div>
        </Card>
      </div>
    );
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 p-6 md:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl md:text-4xl font-serif font-bold text-gray-900 mb-2">
              EndoDetect AI - Radiomics Analysis
            </h1>
            <p className="text-base text-gray-600">
              Upload medical images for radiomics-based endometriosis detection
            </p>
          </div>

          {/* Tabs */}
          <div className="mb-6 border-b border-gray-200">
            <div className="flex space-x-8">
              {[
                { id: 'upload', label: 'Upload', icon: Upload },
                { id: 'outputs', label: 'Outputs', icon: FileText },
              ].map((tabItem) => {
                const Icon = tabItem.icon;
                return (
                  <button
                    key={tabItem.id}
                    onClick={() => navigate(`/dashboard?tab=${tabItem.id}`)}
                    className={`flex items-center gap-2 px-4 py-3 border-b-2 font-medium transition-colors ${
                      tab === tabItem.id
                        ? 'border-primary-600 text-primary-700'
                        : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
                    }`}
                  >
                    <Icon size={18} />
                    {tabItem.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Content */}
          {tab === 'upload' && renderUpload()}
          {tab === 'outputs' && renderOutputs()}
        </div>
      </div>
    </div>
  );
}
