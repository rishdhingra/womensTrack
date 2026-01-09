import Card from './Card';
import { AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';

export default function OutputCards({ outputs }) {
  if (!outputs) return null;

  const getRiskIcon = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'high':
        return <AlertCircle className="text-red-600" size={24} />;
      case 'moderate':
        return <AlertTriangle className="text-yellow-600" size={24} />;
      case 'low':
        return <CheckCircle className="text-green-600" size={24} />;
      default:
        return <Info className="text-blue-600" size={24} />;
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <div className="flex items-start justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Predicted Phenotype</h3>
        </div>
        <p className="text-2xl font-serif font-bold text-primary-700 mb-2">
          {outputs.phenotype || 'N/A'}
        </p>
        <p className="text-sm text-gray-600">
          AI-detected endometriosis subtype classification
        </p>
      </Card>

      <Card>
        <div className="flex items-start justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Confidence</h3>
        </div>
        <div className="mb-4">
          <div className="flex items-baseline space-x-2 mb-2">
            <span className="text-3xl font-bold text-gray-900">
              {outputs.confidence || 0}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-primary-600 h-3 rounded-full transition-all duration-500"
              style={{ width: `${outputs.confidence || 0}%` }}
            />
          </div>
        </div>
        <p className="text-sm text-gray-600">
          Model confidence score
        </p>
      </Card>

      <Card>
        <div className="flex items-start justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Risk / Severity Flag</h3>
          {getRiskIcon(outputs.risk)}
        </div>
        <p className="text-2xl font-serif font-bold text-gray-900 mb-2">
          {outputs.risk || 'N/A'}
        </p>
        <p className="text-sm text-gray-600">
          Stratified risk assessment
        </p>
      </Card>

      <Card>
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Suggested Next Steps</h3>
          <ul className="space-y-2">
            {(outputs.nextSteps || []).map((step, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-primary-600 mt-1">•</span>
                <span className="text-sm text-gray-700">{step}</span>
              </li>
            ))}
          </ul>
        </div>
      </Card>
    </div>
  );
}

