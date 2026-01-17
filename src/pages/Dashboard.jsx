import { useState, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { User, Activity, Microscope, Beaker, Syringe, Calendar } from 'lucide-react';
import Sidebar from '../components/Sidebar';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import SurgicalRoadmap from '../components/SurgicalRoadmap';
import { samplePatients, patientList } from '../data/samplePatients';

export default function Dashboard() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tab = searchParams.get('tab') || 'overview';
  
  const [selectedPatientId, setSelectedPatientId] = useState('die_high_pain');
  
  const selectedPatient = useMemo(() => {
    return samplePatients[selectedPatientId];
  }, [selectedPatientId]);

  const renderPatientSelector = () => (
    <Card className="mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <User size={20} className="text-primary-600" />
          Select Demo Case
        </h3>
        <Badge variant="secondary">Demo Data</Badge>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {patientList.map((patient) => (
          <button
            key={patient.id}
            onClick={() => setSelectedPatientId(patient.id)}
            className={`p-4 rounded-lg border-2 transition-all text-left ${
              selectedPatientId === patient.id
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-200 hover:border-primary-300 bg-white'
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <span className="text-sm font-semibold text-gray-900">{patient.label}</span>
              <Badge variant={selectedPatientId === patient.id ? 'primary' : 'secondary'} className="text-xs">
                {patient.tag}
              </Badge>
            </div>
            <p className="text-xs text-gray-600">Click to view</p>
          </button>
        ))}
      </div>
      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-xs text-blue-800">
          <strong>Data Source:</strong> Demo cases modeled after UT-EndoMRI dataset specifications (Liang et al. 2025, DOI: 10.5281/zenodo.15750762)
        </p>
      </div>
    </Card>
  );

  const renderOverview = () => (
    <div className="space-y-6">
      {renderPatientSelector()}
      
      {/* Patient Summary Card */}
      <Card>
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-2xl font-serif font-bold text-gray-900 mb-2">
              {selectedPatient.name}
            </h2>
            <p className="text-base text-gray-600">{selectedPatient.description}</p>
          </div>
          <Badge variant="primary" className="text-sm">{selectedPatient.cohort}</Badge>
        </div>

        {/* Demographics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div>
            <p className="text-sm text-gray-600">Age</p>
            <p className="text-lg font-semibold text-gray-900">{selectedPatient.demographics.age} years</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">BMI</p>
            <p className="text-lg font-semibold text-gray-900">{selectedPatient.demographics.bmi}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Symptom Duration</p>
            <p className="text-lg font-semibold text-gray-900">{selectedPatient.demographics.symptom_duration_years} years</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Parity</p>
            <p className="text-lg font-semibold text-gray-900">
              {selectedPatient.demographics.nulliparous ? 'Nulliparous' : 'Parous'}
            </p>
          </div>
        </div>

        {/* Pain Scores (WERF EPHect) */}
        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Activity size={18} className="text-primary-600" />
            WERF EPHect Pain Phenotyping
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(selectedPatient.phenotype.pain_scores).map(([type, score]) => (
              <div key={type}>
                <p className="text-sm text-gray-600 mb-2 capitalize">
                  {type.replace(/_/g, ' ')}
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        score >= 7 ? 'bg-red-500' : score >= 4 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${(score / 10) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-gray-900 w-8">{score}/10</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Imaging Results (Aim 1 - Diagnostics) */}
      <Card>
        <h3 className="text-xl font-serif font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Microscope size={20} className="text-primary-600" />
          Imaging Analysis (Aim 1: Diagnostics)
        </h3>
        
        <div className="mb-6">
          <p className="text-sm text-gray-600 mb-2">
            <strong>Modality:</strong> {selectedPatient.imaging.modality}
          </p>
          <p className="text-sm text-gray-600">
            <strong>Acquisition Date:</strong> {selectedPatient.imaging.acquisition_date}
          </p>
          {selectedPatient.imaging.dice_score && (
            <p className="text-sm text-gray-600">
              <strong>Model Dice Score:</strong> {(selectedPatient.imaging.dice_score * 100).toFixed(1)}%
            </p>
          )}
        </div>

        {/* Phenotype Classification */}
        <div className="space-y-4 mb-6">
          <h4 className="font-semibold text-gray-900">Phenotype Classification</h4>
          {Object.entries(selectedPatient.imaging.phenotype_classification).map(([phenotype, data]) => (
            <div key={phenotype}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 capitalize">{phenotype}</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-gray-900">
                    {(data.probability * 100).toFixed(0)}%
                  </span>
                  <Badge 
                    variant={
                      data.confidence === 'very_high' || data.confidence === 'high' ? 'primary' :
                      data.confidence === 'medium' ? 'secondary' : 'default'
                    }
                    className="text-xs"
                  >
                    {data.confidence.replace('_', ' ')}
                  </Badge>
                </div>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500"
                  style={{ width: `${data.probability * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Lesion Locations */}
        {selectedPatient.imaging.lesion_locations.length > 0 && (
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Lesion Locations</h4>
            <div className="space-y-3">
              {selectedPatient.imaging.lesion_locations.map((lesion, idx) => (
                <div key={idx} className="border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">{lesion.organ}</span>
                    <span className="text-sm font-semibold text-primary-700">{lesion.involvement}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden mb-2">
                    <div
                      className="h-full bg-gradient-to-r from-primary-400 to-primary-600"
                      style={{ width: `${lesion.involvement}%` }}
                    />
                  </div>
                  {lesion.depth_mm && (
                    <p className="text-xs text-gray-600">Depth: {lesion.depth_mm}mm</p>
                  )}
                  {lesion.diameter_mm && (
                    <p className="text-xs text-gray-600">Diameter: {lesion.diameter_mm}mm</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedPatient.imaging.notes && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800"><strong>Note:</strong> {selectedPatient.imaging.notes}</p>
          </div>
        )}
      </Card>

      {/* Blood Biomarkers (Aim 2 - Drivers) */}
      <Card>
        <h3 className="text-xl font-serif font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Beaker size={20} className="text-lavender-600" />
          Blood Biomarkers (Aim 2: Drivers)
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="border border-gray-200 rounded-lg p-3">
            <p className="text-sm text-gray-600 mb-1">WBC Count</p>
            <p className="text-lg font-semibold text-gray-900">{selectedPatient.blood_biomarkers.wbc_count.toLocaleString()}</p>
            <p className="text-xs text-gray-500">cells/μL</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-3">
            <p className="text-sm text-gray-600 mb-1">NLR</p>
            <p className={`text-lg font-semibold ${
              selectedPatient.blood_biomarkers.nlr > 3 ? 'text-red-600' : 'text-green-600'
            }`}>
              {selectedPatient.blood_biomarkers.nlr.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500">{selectedPatient.blood_biomarkers.nlr > 3 ? 'Elevated' : 'Normal'}</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-3">
            <p className="text-sm text-gray-600 mb-1">CRP</p>
            <p className={`text-lg font-semibold ${
              selectedPatient.blood_biomarkers.crp_mg_L > 3 ? 'text-red-600' : 'text-green-600'
            }`}>
              {selectedPatient.blood_biomarkers.crp_mg_L} mg/L
            </p>
            <p className="text-xs text-gray-500">{selectedPatient.blood_biomarkers.crp_mg_L > 3 ? 'Elevated' : 'Normal'}</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-3">
            <p className="text-sm text-gray-600 mb-1">ESR</p>
            <p className={`text-lg font-semibold ${
              selectedPatient.blood_biomarkers.esr_mm_hr > 20 ? 'text-red-600' : 'text-green-600'
            }`}>
              {selectedPatient.blood_biomarkers.esr_mm_hr} mm/hr
            </p>
            <p className="text-xs text-gray-500">{selectedPatient.blood_biomarkers.esr_mm_hr > 20 ? 'Elevated' : 'Normal'}</p>
          </div>
        </div>

        <div className="p-4 bg-lavender-50 border border-lavender-200 rounded-lg">
          <p className="font-semibold text-lavender-900 mb-2">Inflammatory Endotype</p>
          <p className="text-sm text-lavender-800">{selectedPatient.blood_biomarkers.inflammatory_endotype}</p>
          {selectedPatient.blood_biomarkers.correlation_strength && (
            <p className="text-xs text-lavender-700 mt-2">
              Imaging-blood correlation: {(selectedPatient.blood_biomarkers.correlation_strength * 100).toFixed(0)}%
            </p>
          )}
          {selectedPatient.blood_biomarkers.notes && (
            <p className="text-xs text-lavender-700 mt-2">
              <strong>Note:</strong> {selectedPatient.blood_biomarkers.notes}
            </p>
          )}
        </div>
      </Card>
    </div>
  );

  const renderSurgicalPlanning = () => (
    <div className="space-y-6">
      {renderPatientSelector()}
      
      <Card>
        <h2 className="text-2xl font-serif font-bold text-gray-900 mb-6">
          Surgical Planning - {selectedPatient.name}
        </h2>
        
        <SurgicalRoadmap 
          complexity={selectedPatient.surgical_roadmap.complexity}
          estimatedTime={selectedPatient.surgical_roadmap.estimated_or_time_min}
          recommendations={selectedPatient.surgical_roadmap.preop_recommendations}
          organInvolvement={selectedPatient.imaging.lesion_locations}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Recommended Team</h4>
            <ul className="space-y-2">
              {selectedPatient.surgical_roadmap.recommended_team.map((member, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                  <span className="text-primary-600 mt-1">•</span>
                  {member}
                </li>
              ))}
            </ul>
          </div>
          
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Surgical Approach</h4>
            <p className="text-sm text-gray-700 mb-4">{selectedPatient.surgical_roadmap.surgical_approach}</p>
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm font-semibold text-yellow-900">Recurrence Risk</p>
              <p className="text-sm text-yellow-800">{selectedPatient.surgical_roadmap.recurrence_risk}</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );

  const renderRadiomics = () => (
    <div className="space-y-6">
      {renderPatientSelector()}
      
      <Card>
        <h2 className="text-2xl font-serif font-bold text-gray-900 mb-6">
          Radiomics Features - {selectedPatient.name}
        </h2>
        
        {/* First Order Features */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-900 mb-3">First Order Statistics</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {Object.entries(selectedPatient.radiomics.first_order).map(([key, value]) => (
              <div key={key} className="border border-gray-200 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1 capitalize">{key.replace(/_/g, ' ')}</p>
                <p className="text-sm font-semibold text-gray-900">{typeof value === 'number' ? value.toFixed(2) : value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Shape Features */}
        {selectedPatient.radiomics.shape && (
          <div className="mb-6">
            <h4 className="font-semibold text-gray-900 mb-3">Shape Features</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(selectedPatient.radiomics.shape).map(([key, value]) => (
                <div key={key} className="border border-gray-200 rounded-lg p-3">
                  <p className="text-xs text-gray-600 mb-1 capitalize">{key.replace(/_/g, ' ')}</p>
                  <p className="text-sm font-semibold text-gray-900">{typeof value === 'number' ? value.toFixed(2) : value}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Texture GLCM */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-900 mb-3">Texture (GLCM)</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(selectedPatient.radiomics.texture_glcm).map(([key, value]) => (
              <div key={key} className="border border-gray-200 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1 capitalize">{key.replace(/_/g, ' ')}</p>
                <p className="text-sm font-semibold text-gray-900">{typeof value === 'number' ? value.toFixed(2) : value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Texture GLRLM */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-900 mb-3">Texture (GLRLM)</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(selectedPatient.radiomics.texture_glrlm).map(([key, value]) => (
              <div key={key} className="border border-gray-200 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1 capitalize">{key.replace(/_/g, ' ')}</p>
                <p className="text-sm font-semibold text-gray-900">{typeof value === 'number' ? value.toFixed(2) : value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Wavelet Features */}
        <div>
          <h4 className="font-semibold text-gray-900 mb-3">Wavelet Features</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(selectedPatient.radiomics.wavelet).map(([key, value]) => (
              <div key={key} className="border border-gray-200 rounded-lg p-3">
                <p className="text-xs text-gray-600 mb-1 uppercase">{key.replace(/_/g, ' ')}</p>
                <p className="text-sm font-semibold text-gray-900">{typeof value === 'number' ? value.toFixed(2) : value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Total Features Extracted:</strong> 100+ radiomics features enable quantitative disease characterization and inflammatory endotype classification.
          </p>
        </div>
      </Card>
    </div>
  );

  const renderBiobank = () => (
    <div className="space-y-6">
      {renderPatientSelector()}
      
      <Card>
        <h2 className="text-2xl font-serif font-bold text-gray-900 mb-6 flex items-center gap-2">
          <Syringe size={24} className="text-primary-600" />
          Biobank Priority (Aim 3: Development)
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="border-l-4 border-primary-500 pl-4">
            <p className="text-sm text-gray-600 mb-2">Enrollment Priority</p>
            <p className="text-2xl font-bold text-primary-700">{selectedPatient.biobank.priority}</p>
          </div>
          <div className="border-l-4 border-lavender-500 pl-4">
            <p className="text-sm text-gray-600 mb-2">Cohort</p>
            <p className="text-lg font-semibold text-gray-900">{selectedPatient.cohort}</p>
          </div>
        </div>

        <div className="mb-6">
          <h4 className="font-semibold text-gray-900 mb-3">Rationale</h4>
          <p className="text-sm text-gray-700 bg-gray-50 p-4 rounded-lg">{selectedPatient.biobank.rationale}</p>
        </div>

        <div>
          <h4 className="font-semibold text-gray-900 mb-3">Samples to Collect</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {selectedPatient.biobank.samples_to_collect.map((sample, idx) => (
              <div key={idx} className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg">
                <Calendar size={16} className="text-primary-600 flex-shrink-0" />
                <span className="text-sm text-gray-700">{sample}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-sm text-green-800 font-semibold mb-2">WERF EPHect Standardized Phenotyping</p>
          <p className="text-sm text-green-700">
            All samples linked to imaging-derived features, inflammatory markers, and standardized clinical phenotyping for non-hormonal therapeutic development.
          </p>
        </div>
      </Card>
    </div>
  );

  const renderContent = () => {
    switch (tab) {
      case 'overview':
        return renderOverview();
      case 'surgical-planning':
        return renderSurgicalPlanning();
      case 'radiomics':
        return renderRadiomics();
      case 'biobank':
        return renderBiobank();
      default:
        return renderOverview();
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 p-6 md:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl md:text-4xl font-serif font-bold text-gray-900 mb-2">
              EndoDetect AI Dashboard
            </h1>
            <p className="text-base text-gray-600">
              D³ Framework: Diagnostics • Drivers • Development
            </p>
          </div>

          {/* Tabs */}
          <div className="mb-6 border-b border-gray-200">
            <div className="flex space-x-8 overflow-x-auto">
              {[
                { id: 'overview', label: 'Overview', icon: Activity },
                { id: 'surgical-planning', label: 'Surgical Planning', icon: Microscope },
                { id: 'radiomics', label: 'Radiomics', icon: Beaker },
                { id: 'biobank', label: 'Biobank', icon: Syringe },
              ].map((tabItem) => {
                const Icon = tabItem.icon;
                return (
                  <button
                    key={tabItem.id}
                    onClick={() => navigate(`/dashboard?tab=${tabItem.id}`)}
                    className={`flex items-center gap-2 px-4 py-3 border-b-2 font-medium transition-colors whitespace-nowrap ${
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
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
