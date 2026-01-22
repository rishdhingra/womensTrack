import Card from './Card';
import { FileText, Clock, Users, AlertCircle } from 'lucide-react';

export default function SurgicalRoadmap({ roadmap }) {
  if (!roadmap) {
    return (
      <Card>
        <div className="text-center py-8">
          <p className="text-gray-500">No surgical roadmap available. Run inference first.</p>
        </div>
      </Card>
    );
  }

  const getComplexityColor = (complexity) => {
    switch (complexity?.toLowerCase()) {
      case 'low':
        return 'text-green-600 bg-green-50';
      case 'moderate':
        return 'text-yellow-600 bg-yellow-50';
      case 'high':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-2xl font-serif font-bold text-gray-900 mb-2">
              Surgical Roadmap
            </h3>
            <p className="text-sm text-gray-600">
              AI-generated pre-operative surgical plan
            </p>
          </div>
          <span className={`px-4 py-2 rounded-full font-semibold ${getComplexityColor(roadmap.complexity)}`}>
            {roadmap.complexity} Complexity
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="flex items-center space-x-3">
            <FileText className="text-primary-600" size={24} />
            <div>
              <p className="text-sm text-gray-600">Phenotype</p>
              <p className="font-semibold text-gray-900">{roadmap.phenotype}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <AlertCircle className="text-primary-600" size={24} />
            <div>
              <p className="text-sm text-gray-600">Confidence</p>
              <p className="font-semibold text-gray-900">{roadmap.confidence}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <FileText className="text-primary-600" size={24} />
            <div>
              <p className="text-sm text-gray-600">Lesion Volume</p>
              <p className="font-semibold text-gray-900">{roadmap.lesion_percentage?.toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Organ Involvement */}
      <Card>
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Organ Involvement</h4>
        <div className="space-y-4">
          {roadmap.organ_involvement && roadmap.organ_involvement.length > 0 ? (
            roadmap.organ_involvement.map((organ, index) => (
              <div key={index} className="border-l-4 border-primary-500 pl-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <p className="font-semibold text-gray-900">{organ.name}</p>
                    <p className="text-sm text-gray-600 capitalize">{organ.side}</p>
                  </div>
                  <span className="text-primary-700 font-bold">{organ.involvement}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${organ.involvement}%` }}
                  />
                </div>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-center py-4">No organ involvement detected</p>
          )}
        </div>
      </Card>

      {/* Surgical Recommendations */}
      <Card>
        <div className="flex items-center space-x-3 mb-4">
          <Users className="text-primary-600" size={24} />
          <h4 className="text-lg font-semibold text-gray-900">Surgical Recommendations</h4>
        </div>
        <ul className="space-y-3">
          {roadmap.recommendations && roadmap.recommendations.length > 0 ? (
            roadmap.recommendations.map((rec, index) => (
              <li key={index} className="flex items-start space-x-3">
                <div className="mt-1 w-2 h-2 bg-primary-600 rounded-full flex-shrink-0" />
                <p className="text-gray-700">{rec}</p>
              </li>
            ))
          ) : (
            <p className="text-gray-500">No specific recommendations available</p>
          )}
        </ul>
      </Card>

      {/* OR Time & Team Requirements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center space-x-3 mb-4">
            <Clock className="text-primary-600" size={24} />
            <h4 className="text-lg font-semibold text-gray-900">Estimated OR Time</h4>
          </div>
          <p className="text-3xl font-bold text-primary-700">
            {roadmap.complexity === 'Low'
              ? '60-90 min'
              : roadmap.complexity === 'Moderate'
              ? '2-3 hours'
              : '3-4 hours'}
          </p>
        </Card>

        <Card>
          <div className="flex items-center space-x-3 mb-4">
            <Users className="text-primary-600" size={24} />
            <h4 className="text-lg font-semibold text-gray-900">Team Requirements</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-primary-600 rounded-full" />
              <span className="text-gray-700">Advanced laparoscopic surgeon</span>
            </li>
            {roadmap.complexity !== 'Low' && (
              <>
                <li className="flex items-center space-x-2">
                  <span className="w-2 h-2 bg-primary-600 rounded-full" />
                  <span className="text-gray-700">Colorectal surgeon (standby)</span>
                </li>
                <li className="flex items-center space-x-2">
                  <span className="w-2 h-2 bg-primary-600 rounded-full" />
                  <span className="text-gray-700">Urology consult available</span>
                </li>
              </>
            )}
          </ul>
        </Card>
      </div>
    </div>
  );
}
