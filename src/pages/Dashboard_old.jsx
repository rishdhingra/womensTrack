import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Loader2, Sparkles, User } from 'lucide-react';
import Sidebar from '../components/Sidebar';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import UploadBox from '../components/UploadBox';
import OutputCards from '../components/OutputCards';
import Charts from '../components/Charts';
import ReportPanel from '../components/ReportPanel';
import SurgicalRoadmap from '../components/SurgicalRoadmap';
import { mockOutputs, mockChartData } from '../data/mockOutputs';
import { samplePatients, patientList } from '../data/samplePatients';

export default function Dashboard() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tab = searchParams.get('tab') || 'overview';
  
  const [mriFile, setMriFile] = useState(null);
  const [tvusFile, setTvusFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [outputs, setOutputs] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const timeoutRef = useRef(null);

  const handleRunInference = useCallback(() => {
    if (!mriFile && !tvusFile) {
      alert('Please upload at least one file');
      return;
    }
    
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    setIsRunning(true);
    timeoutRef.current = setTimeout(() => {
      setOutputs(mockOutputs);
      setCohortData(mockChartData);
      setIsRunning(false);
      timeoutRef.current = null;
    }, 2500);
  }, [mriFile, tvusFile]);

  const handleLoadExample = useCallback(() => {
    setOutputs(mockOutputs);
    setCohortData(mockChartData);
    // Optionally set mock files to show they were "uploaded"
    setMriFile({ name: 'example_mri.dcm' });
    setTvusFile({ name: 'example_tvus.mp4' });
  }, []);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const renderContent = () => {
    switch (tab) {
      case 'upload':
        return (
          <div className="space-y-6">
            <Card>
              <h2 className="text-2xl font-serif font-bold text-gray-900 mb-6">
                Upload Imaging Data
              </h2>
              <div className="space-y-6">
                <UploadBox
                  label="Upload Pelvic MRI (mock)"
                  accept=".dcm,.nii,.zip"
                  onFileSelect={setMriFile}
                  selectedFile={mriFile}
                  onRemove={() => setMriFile(null)}
                />
                <UploadBox
                  label="Upload TVUS (mock)"
                  accept=".mp4,.mov,.png,.jpg"
                  onFileSelect={setTvusFile}
                  selectedFile={tvusFile}
                  onRemove={() => setTvusFile(null)}
                />
                <Button
                  onClick={handleRunInference}
                  disabled={isRunning || (!mriFile && !tvusFile)}
                  variant="primary"
                  className="w-full md:w-auto"
                >
                  {isRunning ? (
                    <>
                      <Loader2 className="inline mr-2 animate-spin" size={20} />
                      Running Demo Inference...
                    </>
                  ) : (
                    'Run Demo Inference'
                  )}
                </Button>
              </div>
            </Card>
          </div>
        );

      case 'outputs':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-serif font-bold text-gray-900">Outputs</h2>
            {outputs ? (
              <>
                <OutputCards outputs={outputs} />
                
                <Card>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Lesion Likelihood Map (Mock)
                  </h3>
                  <div className="bg-gray-100 rounded-lg p-8 mb-4 relative overflow-hidden" style={{ minHeight: '400px' }}>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-64 h-64 mx-auto mb-4 bg-gradient-to-br from-primary-200 to-lavender-200 rounded-lg flex items-center justify-center">
                          <span className="text-gray-600">Image Placeholder</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center space-x-2">
                      <label className="text-sm font-medium text-gray-700">View:</label>
                      <select className="px-3 py-1 border border-gray-300 rounded-lg text-sm">
                        <option>MRI</option>
                        <option>Ultrasound</option>
                      </select>
                    </div>
                    <div className="flex items-center space-x-2 flex-1 min-w-[200px]">
                      <label className="text-sm font-medium text-gray-700">Overlay opacity:</label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        defaultValue="50"
                        className="flex-1"
                      />
                      <span className="text-sm text-gray-600 w-12 text-right">50%</span>
                    </div>
                  </div>
                </Card>
              </>
            ) : (
              <Card>
                <div className="text-center py-12">
                  <p className="text-gray-500 mb-4">No outputs yet. Run inference to see results.</p>
                  <Button
                    onClick={() => navigate('/dashboard?tab=upload')}
                    variant="primary"
                  >
                    Go to Upload
                  </Button>
                </div>
              </Card>
            )}
          </div>
        );

      case 'cohorts':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-serif font-bold text-gray-900">Cohorts</h2>
              {!cohortData && (
                <Button
                  onClick={handleLoadExample}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Sparkles size={18} />
                  Load Example Data
                </Button>
              )}
            </div>
            {cohortData ? (
              <Charts data={cohortData} />
            ) : (
              <Card>
                <div className="text-center py-12">
                  <p className="text-gray-500 mb-4">No cohort data available. Load example data to see cohort analysis.</p>
                  <Button
                    onClick={handleLoadExample}
                    variant="primary"
                    className="flex items-center gap-2 mx-auto"
                  >
                    <Sparkles size={18} />
                    Load Example Data
                  </Button>
                </div>
              </Card>
            )}
          </div>
        );

      case 'reports':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-serif font-bold text-gray-900">Reports</h2>
              {!outputs && (
                <Button
                  onClick={handleLoadExample}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Sparkles size={18} />
                  Load Example Data
                </Button>
              )}
            </div>
            <ReportPanel outputs={outputs} />
          </div>
        );

      default: // overview
        return (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-serif font-bold text-gray-900 mb-2">Overview</h2>
              <p className="text-gray-600">Welcome to the EndoDetect AI Demo Dashboard</p>
            </div>

            {!outputs && (
              <Card>
                <div className="text-center py-8">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Get Started
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Upload imaging data to run a demo inference and see AI outputs, or load example data to explore the dashboard.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                      onClick={handleLoadExample}
                      variant="primary"
                      className="flex items-center justify-center gap-2"
                    >
                      <Sparkles size={18} />
                      Load Example Data
                    </Button>
                    <Button
                      onClick={() => navigate('/dashboard?tab=upload')}
                      variant="secondary"
                    >
                      Go to Upload
                    </Button>
                  </div>
                </div>
              </Card>
            )}

            {outputs && (
              <>
                <OutputCards outputs={outputs} />
                <Charts data={cohortData || mockChartData} />
              </>
            )}
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 md:p-8">
          <div className="mb-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h1 className="text-3xl font-serif font-bold text-gray-900">
                  EndoDetect AI Dashboard
                </h1>
                <p className="text-sm text-gray-500 mt-1">
                  Research Use Only • Proof-of-Concept Demo
                </p>
              </div>
              {!outputs && (
                <Button
                  onClick={handleLoadExample}
                  variant="primary"
                  className="flex items-center gap-2"
                >
                  <Sparkles size={18} />
                  Load Example Data
                </Button>
              )}
            </div>
          </div>
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

