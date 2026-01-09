import { motion } from 'framer-motion';
import { Target, Calendar, CheckCircle } from 'lucide-react';
import Section from '../components/Section';
import SectionTitle from '../components/SectionTitle';
import Card from '../components/Card';

export default function Proposal() {
  const aims = [
    {
      title: 'Aim 1: Multimodal AI Development',
      description: 'Develop and validate a multimodal AI system combining MRI and TVUS for endometriosis detection and classification.',
    },
    {
      title: 'Aim 2: Clinical Validation',
      description: 'Conduct prospective clinical validation studies to assess diagnostic accuracy and clinical utility.',
    },
    {
      title: 'Aim 3: Explainable Outputs',
      description: 'Create interpretable, explainable AI outputs that support clinical decision-making and research.',
    },
  ];

  const timeline = [
    {
      year: 'Year 1',
      milestones: [
        'Data collection and curation',
        'Initial model development',
        'Pilot validation studies',
      ],
    },
    {
      year: 'Year 2',
      milestones: [
        'Model refinement and optimization',
        'Prospective validation cohort',
        'Clinical integration planning',
      ],
    },
    {
      year: 'Year 3',
      milestones: [
        'Large-scale validation',
        'Regulatory submission preparation',
        'Publication and dissemination',
      ],
    },
  ];

  const deliverables = [
    'Validated multimodal AI model for endometriosis detection',
    'Clinical validation study results and publications',
    'Explainable AI outputs with lesion likelihood maps',
    'Structured data resource for research use',
    'Clinical decision support framework',
    'Regulatory pathway documentation',
  ];

  return (
    <div className="min-h-screen">
      <Section className="gradient-bg pt-24 pb-16">
        <div className="container mx-auto px-4 md:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center max-w-3xl mx-auto"
          >
            <h1 className="text-4xl md:text-5xl font-serif font-bold text-gray-900 mb-6">
              Research Proposal
            </h1>
            <p className="text-xl text-gray-700">
              Overview of aims, timeline, and deliverables for the EndoDetect AI project
            </p>
          </motion.div>
        </div>
      </Section>

      {/* Aims */}
      <Section>
        <SectionTitle
          title="Research Aims"
          subtitle="Three primary objectives driving the project"
        />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {aims.map((aim, index) => (
            <motion.div
              key={aim.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card>
                <div className="w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center mb-4">
                  <Target className="text-white" size={24} />
                </div>
                <h3 className="text-lg font-serif font-bold text-gray-900 mb-3">
                  {aim.title}
                </h3>
                <p className="text-gray-600 text-sm">{aim.description}</p>
              </Card>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* Timeline */}
      <Section className="bg-gray-50">
        <SectionTitle
          title="Project Timeline"
          subtitle="Three-year roadmap for development and validation"
        />
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {timeline.map((phase, index) => (
              <motion.div
                key={phase.year}
                initial={{ opacity: 0, x: index % 2 === 0 ? -10 : 10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ delay: index * 0.08, duration: 0.3, ease: "easeOut" }}
              >
                <Card>
                  <div className="flex items-center space-x-3 mb-4">
                    <Calendar className="text-primary-600" size={24} />
                    <h3 className="text-xl font-serif font-bold text-gray-900">
                      {phase.year}
                    </h3>
                  </div>
                  <ul className="space-y-2">
                    {phase.milestones.map((milestone, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={16} />
                        <span className="text-sm text-gray-700">{milestone}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* Deliverables */}
      <Section>
        <SectionTitle
          title="Key Deliverables"
          subtitle="Expected outputs from the research project"
        />
        <div className="max-w-3xl mx-auto">
          <Card>
            <ul className="space-y-3">
              {deliverables.map((item, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-start space-x-3"
                >
                  <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                  <span className="text-gray-700">{item}</span>
                </motion.li>
              ))}
            </ul>
          </Card>
        </div>
      </Section>

      {/* Budget Placeholder */}
      <Section className="bg-gray-50">
        <div className="max-w-2xl mx-auto text-center">
          <Card>
            <h3 className="text-xl font-serif font-bold text-gray-900 mb-4">
              Budget Overview
            </h3>
            <p className="text-gray-600 mb-4">
              The project budget covers personnel, infrastructure, data collection,
              and validation studies across the three-year timeline.
            </p>
            <p className="text-sm text-gray-500 italic">
              Detailed budget breakdown available upon request.
            </p>
          </Card>
        </div>
      </Section>
    </div>
  );
}

