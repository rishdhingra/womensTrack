import { useState } from 'react';
import Card from './Card';
import Button from './Button';
import { FileText, Download } from 'lucide-react';

export default function ReportPanel({ outputs }) {
  const [reportGenerated, setReportGenerated] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateReport = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
      setReportGenerated(true);
    }, 2000);
  };

  if (!outputs) {
    return (
      <Card>
        <div className="text-center py-8">
          <FileText size={48} className="mx-auto mb-4 text-gray-300" />
          <p className="text-gray-500">Run inference to generate report</p>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Clinician Summary</h3>
        <Button
          onClick={generateReport}
          disabled={isGenerating || reportGenerated}
          variant="primary"
        >
          {isGenerating ? 'Generating...' : reportGenerated ? 'Regenerate' : 'Generate Report'}
        </Button>
      </div>

      {isGenerating && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mb-4"></div>
          <p className="text-gray-600">Generating report...</p>
        </div>
      )}

      {reportGenerated && !isGenerating && (
        <div className="bg-gray-50 rounded-lg p-6 space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-gray-900">AI Analysis Report</h4>
              <button className="flex items-center space-x-2 text-sm text-primary-600 hover:text-primary-700">
                <Download size={16} />
                <span>Download PDF</span>
              </button>
            </div>

            <div className="space-y-4 text-sm">
              <div>
                <p className="font-semibold text-gray-700 mb-1">Predicted Phenotype:</p>
                <p className="text-gray-900">{outputs.phenotype}</p>
              </div>

              <div>
                <p className="font-semibold text-gray-700 mb-1">Confidence Score:</p>
                <p className="text-gray-900">{outputs.confidence}%</p>
              </div>

              <div>
                <p className="font-semibold text-gray-700 mb-1">Risk Stratification:</p>
                <p className="text-gray-900">{outputs.risk}</p>
              </div>

              <div>
                <p className="font-semibold text-gray-700 mb-1">Interpretability:</p>
                <p className="text-gray-600">
                  This analysis uses explainable AI outputs, including lesion likelihood maps
                  and feature importance scores. The model combines radiomics features from
                  MRI and computer vision features from ultrasound to provide a comprehensive
                  assessment.
                </p>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <p className="text-xs text-gray-500 italic">
                  <strong>Disclaimer:</strong> This is a research prototype. Results are for
                  research use only and should not be used for clinical decision-making without
                  proper validation and regulatory approval.
                </p>
              </div>
            </div>
        </div>
      )}
    </Card>
  );
}

