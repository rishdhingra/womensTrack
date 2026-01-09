import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Mail, ExternalLink } from 'lucide-react';
import rwjLogo from '../assets/rwj.png';
import ucsfLogo from '../assets/ucsf.png';
import awsLogo from '../assets/aws.png';

export default function Footer() {
  const location = useLocation();
  const navigate = useNavigate();

  const handleHowItWorksClick = (e) => {
    e.preventDefault();
    if (location.pathname === '/') {
      // Already on landing page, scroll to section
      const element = document.getElementById('how-it-works');
      if (element) {
        const offset = 80; // Account for sticky navbar
        const elementPosition = element.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - offset;
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    } else {
      // Navigate to landing page then scroll
      navigate('/');
      // Wait for page to load and render
      setTimeout(() => {
        const element = document.getElementById('how-it-works');
        if (element) {
          const offset = 80;
          const elementPosition = element.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - offset;
          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });
        } else {
          // If element not found, try again after a longer delay
          setTimeout(() => {
            const retryElement = document.getElementById('how-it-works');
            if (retryElement) {
              const offset = 80;
              const elementPosition = retryElement.getBoundingClientRect().top;
              const offsetPosition = elementPosition + window.pageYOffset - offset;
              window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
              });
            }
          }, 300);
        }
      }, 150);
    }
  };

  const handleDashboardClick = (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    navigate('/dashboard');
  };

  return (
    <footer className="bg-gray-50 border-t border-gray-200">
      <div className="container mx-auto px-4 md:px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <h3 className="text-lg font-serif font-bold text-gray-900 mb-4">
              EndoDetect AI
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              A research and decision-support platform using AI to analyze pelvic medical imaging for endometriosis characterization.
            </p>
            <p className="text-xs text-gray-500 italic">
              Proof-of-concept platform — not intended for clinical diagnosis or treatment.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <a 
                  href="/dashboard" 
                  onClick={handleDashboardClick}
                  className="text-sm text-gray-600 hover:text-primary-600 cursor-pointer"
                >
                  Demo Dashboard
                </a>
              </li>
              <li>
                <a 
                  href="/#how-it-works" 
                  onClick={handleHowItWorksClick}
                  className="text-sm text-gray-600 hover:text-primary-600 cursor-pointer"
                >
                  How It Works
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-4">Contact</h4>
            <a
              href="mailto:contact@rutgers.edu"
              className="flex items-center space-x-2 text-sm text-gray-600 hover:text-primary-600"
            >
              <Mail size={16} />
              <span>Contact Us</span>
            </a>
          </div>
        </div>

        {/* Sponsors Strip */}
        <div className="border-t border-gray-200 pt-8">
          <div className="flex flex-wrap items-center justify-center gap-8 mb-4">
            <div className="h-12 flex items-center justify-center">
              <img src={rwjLogo} alt="RWJMS" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </div>
            <div className="h-12 flex items-center justify-center">
              <img src={ucsfLogo} alt="UCSF" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </div>
            <div className="h-12 flex items-center justify-center">
              <img src={awsLogo} alt="AWS" className="h-full w-auto object-contain opacity-80 hover:opacity-100 transition-opacity" />
            </div>
          </div>
        </div>

        <div className="border-t border-gray-200 pt-6 mt-6">
          <p className="text-xs text-center text-gray-500">
            © {new Date().getFullYear()} EndoDetect AI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}

