import { motion } from 'framer-motion';
import Section from '../components/Section';
import SectionTitle from '../components/SectionTitle';
import Card from '../components/Card';

export default function Pipeline() {
  const pipelineSteps = [
    {
      id: 1,
      title: 'Ingest',
      description: 'Pelvic MRI, TVUS, clinical metadata, and routine laboratory markers.',
      details: 'DICOM files from pelvic MRI and transvaginal ultrasound are ingested and validated, along with clinical metadata and laboratory markers.',
    },
    {
      id: 2,
      title: 'Preprocessing',
      description: 'DICOM normalization, quality control, and harmonization.',
      details: 'Image normalization, bias field correction, slice alignment, resolution standardization, and quality control checks.',
    },
    {
      id: 3,
      title: 'Feature Extraction',
      description: 'Radiomics features and deep learning (CNN) embeddings.',
      details: 'Radiomics: texture, shape, intensity features. CNN: deep learning features from pre-trained models for comprehensive feature extraction.',
    },
    {
      id: 4,
      title: 'Model Outputs',
      description: 'Endometriosis phenotype probabilities, lesion localization maps, quantitative risk and inflammation-associated scores.',
      details: 'Ensemble model combining MRI and ultrasound features generates phenotype probabilities, lesion maps, and quantitative risk scores.',
    },
    {
      id: 5,
      title: 'Clinical & Research Use',
      description: 'Pre-operative planning support, patient stratification for research studies, imaging-biomarker discovery.',
      details: 'Explainable, structured outputs formatted for clinical review, decision support, and research applications.',
    },
  ];

  return (
    <div className="min-h-screen">
      <Section className="gradient-bg pt-24 pb-16">
        <div className="container mx-auto px-4 md:px-6">
          <motion.div
            className="text-center max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-5xl font-serif font-bold text-gray-900 mb-6">
              How EndoDetect AI Works
            </h1>
            <p className="text-xl text-gray-700">
              A comprehensive pipeline transforming multimodal imaging into actionable clinical insights
            </p>
          </motion.div>
        </div>
      </Section>

      {/* Pipeline Diagram */}
      <Section>
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-12">
            {pipelineSteps.map((step, index) => {
              return (
                <motion.div
                  key={step.id}
                  className="relative"
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="bg-white rounded-xl border-2 border-primary-100 p-6 h-full hover:border-primary-300 hover:shadow-lg transition-all duration-300">
                    <div className="flex flex-col items-center text-center">
                      <div className="w-14 h-14 bg-gradient-to-br from-primary-500 to-primary-700 text-white rounded-xl flex items-center justify-center mb-4 text-lg font-bold shadow-md">
                        {step.id}
                      </div>
                      <h3 className="text-lg font-serif font-bold text-gray-900 mb-3">{step.title}</h3>
                      <p className="text-sm text-gray-600 leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                  {index < pipelineSteps.length - 1 && (
                    <div className="hidden md:block absolute top-1/2 left-full w-full h-0.5 bg-gradient-to-r from-primary-200 to-primary-100 transform -translate-y-1/2 z-0">
                      <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-3 h-3 bg-primary-400 rounded-full border-2 border-white"></div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>
      </Section>

      {/* Detailed Explanations */}
      <Section className="bg-gray-50">
        <div className="max-w-4xl mx-auto space-y-8">
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <h3 className="text-2xl font-serif font-bold text-gray-900 mb-4">
                What Radiomics Means
              </h3>
              <p className="text-gray-700 leading-relaxed">
                Radiomics extracts quantitative features from medical images that capture texture,
                shape, intensity patterns, and spatial relationships. These features are often
                invisible to the human eye but can reveal important characteristics of tissue
                pathology. For endometriosis, radiomics can identify subtle patterns in MRI
                that correlate with disease presence and subtype.
              </p>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <h3 className="text-2xl font-serif font-bold text-gray-900 mb-4">
                What CNN Means
              </h3>
              <p className="text-gray-700 leading-relaxed">
                Convolutional Neural Networks (CNNs) are deep learning models that automatically
                learn hierarchical features from images. Unlike radiomics which uses predefined
                feature calculations, CNNs discover patterns through training on large datasets.
                Our CNN models are trained to recognize endometriosis-related patterns in both
                MRI and ultrasound images, complementing radiomics features for a comprehensive
                analysis.
              </p>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <h3 className="text-2xl font-serif font-bold text-gray-900 mb-4">
                Why MRI + Ultrasound
              </h3>
              <p className="text-gray-700 leading-relaxed">
                Combining MRI and transvaginal ultrasound (TVUS) provides complementary information
                for endometriosis detection. MRI offers excellent soft tissue contrast and can
                visualize deep infiltrating endometriosis, while TVUS provides real-time imaging
                with high resolution for superficial lesions and ovarian endometriomas. Our
                multimodal approach leverages the strengths of both imaging modalities to improve
                diagnostic accuracy and provide more comprehensive disease characterization.
              </p>
            </Card>
          </motion.div>
        </div>
      </Section>

      {/* Tech Stack */}
      <Section>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="Technical Stack"
            subtitle="Technologies powering the EndoDetect AI pipeline"
          />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6 }}
        >
          <Card>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <h4 className="font-semibold text-gray-900 mb-3">Core Technologies</h4>
                <ul className="space-y-2 text-gray-700">
                  <li>• Python (primary language)</li>
                  <li>• PyTorch / MONAI (deep learning)</li>
                  <li>• nnU-Net (future segmentation)</li>
                </ul>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <h4 className="font-semibold text-gray-900 mb-3">Infrastructure</h4>
                <ul className="space-y-2 text-gray-700">
                  <li>• AWS S3 (data storage)</li>
                  <li>• AWS EC2 / SageMaker (training)</li>
                  <li>• DICOM processing libraries</li>
                </ul>
              </motion.div>
            </div>
          </Card>
        </motion.div>
      </Section>
    </div>
  );
}

