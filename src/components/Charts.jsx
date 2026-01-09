import { useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Card from './Card';

export default function Charts({ data }) {
  const featureData = useMemo(() => data?.features || [
    { name: 'Texture', value: 0.85 },
    { name: 'Shape', value: 0.72 },
    { name: 'Intensity', value: 0.68 },
    { name: 'Gradient', value: 0.61 },
    { name: 'Wavelet', value: 0.54 },
  ], [data?.features]);

  const cohortData = useMemo(() => data?.cohort || [
    { name: 'Q1', Current: 0.65, Reference: 0.58 },
    { name: 'Q2', Current: 0.72, Reference: 0.61 },
    { name: 'Q3', Current: 0.78, Reference: 0.64 },
    { name: 'Q4', Current: 0.82, Reference: 0.67 },
  ], [data?.cohort]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Feature Summary (Mock)
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={featureData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#ec4899" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Cohort Comparison (Mock)
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={cohortData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="Current" stroke="#ec4899" strokeWidth={2} />
            <Line type="monotone" dataKey="Reference" stroke="#8b5cf6" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}

