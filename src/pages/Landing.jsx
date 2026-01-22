import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle, Upload, Sparkles, FileText } from 'lucide-react';
import { useMemo } from 'react';
import { motion } from 'framer-motion';
import Section from '../components/Section';
import SectionTitle from '../components/SectionTitle';
import Card from '../components/Card';
import Button from '../components/Button';
import Badge from '../components/Badge';
import AnimatedWomenSVG from '../components/AnimatedWomenSVG';
import rwjLogo from '../assets/rwj.png';
import ucsfLogo from '../assets/ucsf.png';
import awsLogo from '../assets/aws.png';

export default function Landing() {
  const problemCards = useMemo(() => [
    {
      title: 'Operator-Dependent Interpretation',
      description: 'Interpretation of pelvic imaging is operator-dependent, leading to variability in diagnosis and staging.',
    },
    {
      title: 'Phenotypes Under-Recognized',
      description: 'Different endometriosis phenotypes (DIE, ovarian endometrioma, superficial disease) are often under-recognized on imaging.',
    },
    {
      title: 'Diagnostic Delay',
      description: 'Average diagnostic delay remains 7–10 years, contributing to disease progression and morbidity.',
    },
  ], []);

  const deliverables = useMemo(() => [
    {
      title: 'Automated Phenotype Classification',
      description: 'Classification of endometriosis subtypes (DIE, ovarian endometrioma, superficial disease).',
    },
    {
      title: 'Lesion Likelihood Maps',
      description: 'Spatial probability maps showing disease involvement for visual review.',
    },
    {
      title: 'Quantitative Severity & Risk Stratification',
      description: 'Risk stratification scores to support clinical decision-making.',
    },
    {
      title: 'Pre-Operative Decision-Support Flags',
      description: 'Highlights regions of interest for clinical review and surgical planning.',
    },
    {
      title: 'Explainable AI Outputs',
      description: 'Visual overlays and confidence estimates designed to support clinician judgment.',
    },
    {
      title: 'Research-Ready Outputs',
      description: 'Structured outputs aligned with reporting frameworks for research studies.',
    },
  ], []);

  const easySteps = useMemo(() => [
    {
      icon: Upload,
      title: 'Upload Your Images',
      description: 'Simply upload your pelvic MRI or ultrasound images. No technical knowledge needed—just drag and drop your files.',
      color: 'from-primary-400 to-primary-500',
    },
    {
      icon: Sparkles,
      title: 'AI Does the Work',
      description: 'Our advanced AI automatically analyzes your images. Sit back and relax—the analysis happens in seconds.',
      color: 'from-lavender-400 to-lavender-500',
    },
    {
      icon: FileText,
      title: 'Get Clear Results',
      description: 'Receive easy-to-understand results with visual maps and clear explanations. Everything is designed to be simple and helpful.',
      color: 'from-primary-500 to-lavender-500',
    },
  ], []);

  const steps = useMemo(() => [
    { id: 1, title: 'Ingest', description: 'Pelvic MRI, TVUS, clinical metadata, and routine laboratory markers' },
    { id: 2, title: 'Preprocessing', description: 'DICOM normalization, quality control, and harmonization' },
    { id: 3, title: 'Feature Extraction', description: 'Radiomics features and deep learning (CNN) embeddings' },
    { id: 4, title: 'Model Outputs', description: 'Phenotype probabilities, lesion maps, risk scores, and clinical flags' },
    { id: 5, title: 'Clinical & Research Use', description: 'Pre-operative planning, patient stratification, imaging-biomarker discovery' },
  ], []);

  return (
    <>
      {/* Hero Section */}
      <Section className="gradient-bg min-h-screen flex items-center">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 items-center w-full">
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <Badge variant="proof">Proof-of-Concept</Badge>
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-serif font-bold text-gray-900 mb-4">
              EndoDetect AI
            </h1>
            <p className="text-lg md:text-xl text-gray-700 mb-3 font-medium">
              A research and decision-support platform using AI to analyze pelvic medical imaging
            </p>
            <p className="text-base md:text-lg text-gray-600 mb-6">
              Supports earlier, more consistent characterization of endometriosis through quantitative imaging analysis.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Link to="/dashboard">
                <Button variant="primary" className="w-full sm:w-auto text-base">
                  Open Demo Dashboard
                  <ArrowRight className="inline ml-2" size={18} />
                </Button>
              </Link>
              <Link to="/pipeline">
                <Button variant="secondary" className="w-full sm:w-auto text-base">
                  How It Works
                </Button>
              </Link>
            </div>
          </div>
          <div className="hidden md:block">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-primary-200 to-lavender-200 rounded-3xl transform rotate-6"></div>
              <div className="relative bg-white/80 backdrop-blur-sm rounded-3xl p-6 md:p-8 shadow-2xl">
                <div className="w-full h-full min-h-[350px] md:min-h-[400px] flex items-center justify-center">
                  <AnimatedWomenSVG />
                </div>
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* How Easy It Is Section */}
      <Section className="bg-white">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="How Easy It Is"
            subtitle="Three simple steps to get started—no technical expertise required"
          />
        </motion.div>
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {easySteps.map((step, index) => {
              const IconComponent = step.icon;
              return (
                <motion.div
                  key={index}
                  className="relative"
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <Card className="h-full text-center p-6 md:p-8 hover:shadow-xl transition-all duration-300 border-2 border-transparent hover:border-primary-100">
                    <div className="flex flex-col items-center">
                      <div className={`w-20 h-20 md:w-24 md:h-24 bg-gradient-to-br ${step.color} rounded-2xl flex items-center justify-center mb-6 shadow-lg transform hover:scale-110 transition-transform duration-300`}>
                        <IconComponent className="text-white" size={40} strokeWidth={2} />
                      </div>
                      <div className="absolute top-0 right-0 w-8 h-8 md:w-10 md:h-10 bg-gradient-to-br from-primary-200 to-lavender-200 rounded-full flex items-center justify-center text-sm md:text-base font-bold text-primary-700 shadow-md">
                        {index + 1}
                      </div>
                      <h3 className="text-xl md:text-2xl font-serif font-bold text-gray-900 mb-4">
                        {step.title}
                      </h3>
                      <p className="text-base md:text-lg text-gray-600 leading-relaxed">
                        {step.description}
                      </p>
                    </div>
                  </Card>
                  {index < easySteps.length - 1 && (
                    <div className="hidden md:block absolute top-1/2 left-full w-full transform -translate-y-1/2 z-0">
                      <div className="flex items-center justify-center">
                        <ArrowRight className="text-primary-300" size={32} />
                      </div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
          <motion.div
            className="mt-12 text-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <p className="text-base md:text-lg text-gray-600 mb-6 max-w-2xl mx-auto">
              Our platform is designed with you in mind. We believe healthcare technology should be accessible, clear, and supportive—not complicated or intimidating.
            </p>
            <Link to="/dashboard">
              <Button variant="primary" className="text-base md:text-lg px-6 md:px-8 py-3 md:py-4">
                Try It Now
                <ArrowRight className="inline ml-2" size={20} />
              </Button>
            </Link>
          </motion.div>
        </div>
      </Section>

      {/* Problem Section */}
      <Section id="problem" className="gradient-bg">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="The Challenge"
            subtitle="Current limitations in endometriosis imaging assessment"
          />
        </motion.div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {problemCards.map((card, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="h-full">
                <h3 className="text-lg md:text-xl font-serif font-bold text-gray-900 mb-3">
                  {card.title}
                </h3>
                <p className="text-sm md:text-base text-gray-600">{card.description}</p>
              </Card>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* Solution Section */}
      <Section className="gradient-bg">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <SectionTitle
              title="Our Solution"
              subtitle="AI-powered quantitative imaging analysis"
              className="text-left mb-8"
            />
            <ul className="space-y-4 md:space-y-5">
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">Multimodal Analysis</p>
                  <p className="text-sm md:text-base text-gray-600">Pelvic MRI and Transvaginal Ultrasound (TVUS)</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">AI Model Application</p>
                  <p className="text-sm md:text-base text-gray-600">Radiomics features and deep learning (CNN) embeddings</p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">Explainable Outputs</p>
                  <p className="text-sm md:text-base text-gray-600">Designed to support, not replace, clinician judgment</p>
                </div>
              </li>
            </ul>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <Card className="h-full">
              <div className="p-5 md:p-6">
                <h4 className="text-base md:text-lg font-semibold text-gray-900 mb-4">AI Output Preview</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Phenotype</span>
                    <span className="text-sm md:text-base font-semibold text-primary-700">DIE</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Confidence</span>
                    <span className="text-sm md:text-base font-semibold text-primary-700">87%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Risk Level</span>
                    <span className="text-sm md:text-base font-semibold text-primary-700">Moderate</span>
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs text-gray-500">
                      Explainable outputs with lesion likelihood maps
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      </Section>

      {/* How It Works Section */}
      <Section id="how-it-works" className="bg-white">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="How It Works"
            subtitle="End-to-end AI pipeline for endometriosis characterization"
          />
        </motion.div>
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
            {steps.map((step, index) => (
              <motion.div
                key={step.id}
                className="relative"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="h-full hover:shadow-xl transition-shadow duration-300">
                  <div className="flex flex-col items-center text-center p-4">
                    <div className="relative mb-4 md:mb-6">
                      <div className="w-16 h-16 md:w-20 md:h-20 bg-gradient-to-br from-primary-500 via-primary-600 to-primary-700 rounded-2xl flex items-center justify-center text-xl md:text-2xl font-bold text-white shadow-lg transform rotate-3 hover:rotate-0 transition-transform duration-300">
                        {step.id}
                      </div>
                      <div className="absolute -top-1 -right-1 w-5 h-5 md:w-6 md:h-6 bg-lavender-400 rounded-full opacity-80"></div>
                    </div>
                    <h4 className="text-lg md:text-xl font-serif font-bold text-gray-900 mb-2 md:mb-3">{step.title}</h4>
                    <p className="text-xs md:text-sm text-gray-600 leading-relaxed">{step.description}</p>
                  </div>
                </Card>
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 left-full w-full h-1 bg-gradient-to-r from-primary-300 via-primary-200 to-transparent transform -translate-y-1/2 z-0">
                    <div className="absolute right-0 top-1/2 transform -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-primary-500 rounded-full border-4 border-white shadow-md"></div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* Deliverables Section */}
      <Section className="gradient-bg" id="deliverables">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="Deliverables"
            subtitle="Comprehensive AI outputs for clinical and research use"
          />
        </motion.div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {deliverables.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
            >
              <Card className="h-full">
                <h3 className="text-base md:text-lg font-serif font-bold text-gray-900 mb-2 md:mb-3">
                  {item.title}
                </h3>
                <p className="text-sm md:text-base text-gray-600">{item.description}</p>
              </Card>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* D³ Framework Section */}
      <Section className="bg-white">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="The D³ Framework"
            subtitle="Diagnostics • Drivers • Development"
          />
        </motion.div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <Card className="h-full">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center text-white font-bold text-xl mr-4">1</div>
                <h3 className="text-xl font-serif font-bold text-gray-900">Diagnostics</h3>
              </div>
              <p className="text-base text-gray-600 mb-4">
                Multimodal ML model using ~400 patients for non-invasive diagnosis and phenotype classification (DIE, ovarian, superficial).
              </p>
              <p className="text-sm text-primary-700 font-semibold">Aim 1: Radiomics-Based Diagnosis</p>
            </Card>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card className="h-full">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-lavender-500 rounded-full flex items-center justify-center text-white font-bold text-xl mr-4">2</div>
                <h3 className="text-xl font-serif font-bold text-gray-900">Drivers</h3>
              </div>
              <p className="text-base text-gray-600 mb-4">
                Explainable AI to identify inflammatory and neuroimmune drivers of pain, systemic inflammation, and recurrence.
              </p>
              <p className="text-sm text-lavender-700 font-semibold">Aim 2: Inflammatory Endotypes</p>
            </Card>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="h-full">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center text-white font-bold text-xl mr-4">3</div>
                <h3 className="text-xl font-serif font-bold text-gray-900">Development</h3>
              </div>
              <p className="text-base text-gray-600 mb-4">
                Prospective validation (~100 patients) and biobank creation for non-hormonal therapeutic development.
              </p>
              <p className="text-sm text-primary-700 font-semibold">Aim 3: Biobank & Validation</p>
            </Card>
          </motion.div>
        </div>
      </Section>

      {/* Team Section */}
      <Section className="gradient-bg">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="Research Team"
            subtitle="Rutgers Robert Wood Johnson Medical School"
          />
        </motion.div>
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <Card className="mb-8">
              <h4 className="text-lg font-serif font-bold text-primary-700 mb-4">Principal Investigator</h4>
              <p className="text-base text-gray-900 font-semibold">Jessica Opoku-Anane, MD, MS</p>
              <p className="text-sm text-gray-600">Minimally Invasive Gynecologic Surgery & Reproductive Endocrinology</p>
            </Card>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <h4 className="text-lg font-serif font-bold text-primary-700 mb-4">Co-Investigators</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <p className="text-base text-gray-900 font-semibold">Archana Pradhan, MD, MPH</p>
                  <p className="text-sm text-gray-600">Interim Chair, Dept of OBGYN&RS; Associate Dean for Clinical Education</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Naveena Yanamala, MS, PhD</p>
                  <p className="text-sm text-gray-600">Section Chief of Clinical Research & AI Innovation; Director of Data Science and ML Research</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Susan Egan, BS, RDMS</p>
                  <p className="text-sm text-gray-600">Chief Gynecologic Ultrasonographer; Co-author of SRU Consensus on Pelvic US for Endometriosis</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Alopi Patel, MD</p>
                  <p className="text-sm text-gray-600">Interventional Pain Management, Associate Professor, Dept of OBGYN&RS and Anesthesia</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Traci Ito, MD</p>
                  <p className="text-sm text-gray-600">Minimally Invasive Gynecologic Surgery, Associate Professor, UCSF</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Azra Bano</p>
                  <p className="text-sm text-gray-600">Computer Engineering & Data Science, Rutgers School of Engineering</p>
                </div>
                <div>
                  <p className="text-base text-gray-900 font-semibold">Rishabh Dhingra</p>
                  <p className="text-sm text-gray-600">Computer Science, Rutgers School of Arts & Sciences</p>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      </Section>

      {/* Impact Section */}
      <Section id="impact" className="bg-white">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <SectionTitle
            title="Clinical & Research Applications"
            subtitle="Supporting endometriosis care through AI"
          />
        </motion.div>
        <motion.div
          className="max-w-3xl mx-auto"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6 }}
        >
          <Card>
            <ul className="space-y-4 md:space-y-5">
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">Pre-Operative Planning Support</p>
                  <p className="text-sm md:text-base text-gray-600">
                    Decision-support flags and lesion maps aid in surgical planning and patient counseling.
                  </p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">Patient Stratification for Research</p>
                  <p className="text-sm md:text-base text-gray-600">
                    Quantitative scores enable consistent patient enrollment in research studies and clinical trials.
                  </p>
                </div>
              </li>
              <li className="flex items-start space-x-3">
                <CheckCircle className="text-primary-600 mt-1 flex-shrink-0" size={20} />
                <div>
                  <p className="text-sm md:text-base font-semibold text-gray-900 mb-1">Imaging-Biomarker Discovery</p>
                  <p className="text-sm md:text-base text-gray-600">
                    Structured outputs facilitate discovery of novel imaging biomarkers for endometriosis.
                  </p>
                </div>
              </li>
            </ul>
          </Card>
        </motion.div>
      </Section>

      {/* Sponsors Section */}
      <Section className="gradient-bg">
        <motion.div
          className="flex flex-col items-center justify-center gap-6 max-w-4xl mx-auto"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-serif font-bold text-gray-900">
            Supported By:
          </h2>
          <div className="flex items-center justify-center gap-12 md:gap-16 flex-wrap">
            {/* RWJ Logo */}
            <motion.div
              className="h-16 md:h-20 flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <img src={rwjLogo} alt="RWJMS" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </motion.div>
            {/* UCSF Logo */}
            <motion.div
              className="h-16 md:h-20 flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <img src={ucsfLogo} alt="UCSF" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </motion.div>
            {/* AWS Logo */}
            <motion.div
              className="h-16 md:h-20 flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.4, delay: 0.3 }}
            >
              <img src={awsLogo} alt="AWS" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </motion.div>
          </div>
        </motion.div>
      </Section>

      {/* Final CTA */}
      <Section className="bg-white">
        <motion.div
          className="text-center max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl md:text-3xl lg:text-4xl font-serif font-bold text-gray-900 mb-4 md:mb-6">
            See the Demo
          </h2>
          <p className="text-base md:text-lg text-gray-600 mb-6 md:mb-8">
            Explore the EndoDetect AI dashboard and experience the future of endometriosis imaging.
          </p>
          <Link to="/dashboard">
            <Button variant="primary" className="text-base md:text-lg px-6 md:px-8 py-3 md:py-4">
              Open Demo Dashboard
              <ArrowRight className="inline ml-2" size={20} />
            </Button>
          </Link>
        </motion.div>
      </Section>
    </>
  );
}

