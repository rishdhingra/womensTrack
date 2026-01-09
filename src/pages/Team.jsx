import { motion } from 'framer-motion';
import Section from '../components/Section';
import SectionTitle from '../components/SectionTitle';
import Card from '../components/Card';
import Badge from '../components/Badge';

export default function Team() {
  const teamMembers = [
    {
      name: 'Jessica Opoku-Anane',
      role: 'Principal Investigator',
      tag: 'PI',
      description: 'Clinical leadership and research direction',
    },
    {
      name: 'Archana Pradhan',
      role: 'Clinical Leadership',
      tag: 'Clinical',
      description: 'Clinical data curation and validation',
    },
    {
      name: 'Naveena Yanamala',
      role: 'AI/ML Research',
      tag: 'ML',
      description: 'Machine learning pipeline and modeling',
    },
    {
      name: 'Susan Egan',
      role: 'Ultrasound Expertise',
      tag: 'Clinical',
      description: 'Ultrasound imaging and interpretation',
    },
    {
      name: 'Azra Bano',
      role: 'Engineering',
      tag: 'Engineering',
      description: 'ML pipeline development and infrastructure',
    },
    {
      name: 'Rishabh Dhingra',
      role: 'Engineering',
      tag: 'Engineering',
      description: 'UI/dashboard and reporting systems',
    },
    {
      name: 'Alopi Patel',
      role: 'Pain / Clinical',
      tag: 'Clinical',
      description: 'Clinical validation and pain assessment',
    },
    {
      name: 'Traci Ito',
      role: 'UCSF Collaborator',
      tag: 'Collaborator',
      description: 'External validation and research collaboration',
    },
  ];

  const responsibilities = [
    {
      title: 'Clinical Data + Validation',
      items: [
        'Data collection and curation',
        'Clinical validation studies',
        'Expert annotation and ground truth',
        'Regulatory pathway planning',
      ],
    },
    {
      title: 'ML Pipeline + Modeling',
      items: [
        'Feature extraction (radiomics + CNN)',
        'Model training and optimization',
        'Multimodal fusion strategies',
        'Performance evaluation and benchmarking',
      ],
    },
    {
      title: 'UI/Dashboard + Reporting',
      items: [
        'User interface development',
        'Visualization and explainability',
        'Report generation systems',
        'Integration with clinical workflows',
      ],
    },
  ];

  const getTagVariant = (tag) => {
    switch (tag) {
      case 'PI':
        return 'proof';
      case 'Clinical':
        return 'info';
      case 'ML':
        return 'default';
      case 'Engineering':
        return 'success';
      default:
        return 'default';
    }
  };

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
              Our Team
            </h1>
            <p className="text-xl text-gray-700">
              Multidisciplinary expertise in clinical care, AI research, and engineering
            </p>
          </motion.div>
        </div>
      </Section>

      <Section>
        <SectionTitle
          title="Team Members"
          subtitle="Dedicated researchers and clinicians advancing endometriosis care"
        />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, index) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: index * 0.03, duration: 0.3, ease: "easeOut" }}
            >
              <Card>
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-br from-primary-200 to-lavender-200 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <span className="text-2xl font-serif font-bold text-primary-700">
                      {member.name.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>
                  <h3 className="font-serif font-bold text-gray-900 mb-1">
                    {member.name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">{member.role}</p>
                  <Badge variant={getTagVariant(member.tag)} className="mb-3">
                    {member.tag}
                  </Badge>
                  <p className="text-xs text-gray-500">{member.description}</p>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </Section>

      <Section className="bg-gray-50">
        <SectionTitle
          title="Team Responsibilities"
          subtitle="Organized expertise across clinical, ML, and engineering domains"
        />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {responsibilities.map((resp, index) => (
            <motion.div
              key={resp.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card>
                <h3 className="text-xl font-serif font-bold text-gray-900 mb-4">
                  {resp.title}
                </h3>
                <ul className="space-y-2">
                  {resp.items.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-primary-600 mt-1">•</span>
                      <span className="text-sm text-gray-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </motion.div>
          ))}
        </div>
      </Section>
    </div>
  );
}

