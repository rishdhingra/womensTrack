import { useLocation, useNavigate } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import endoheartLogo from '../assets/endoheart.png';

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const navLinks = [
    { to: '/', label: 'Home' },
    { to: '/pipeline', label: 'Pipeline' },
    { to: '/dashboard', label: 'Demo' },
  ];

  const isActive = (path) => location.pathname === path;

  const handleNavClick = (to, e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    navigate(to);
    setMobileMenuOpen(false);
  };

  return (
    <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex items-center justify-between h-14 md:h-16 py-2">
          <a 
            href="/" 
            onClick={(e) => handleNavClick('/', e)}
            className="flex items-center -my-2 md:-my-3 cursor-pointer"
          >
            <img 
              src={endoheartLogo} 
              alt="EndoDetect AI" 
              className="h-40 md:h-48 w-auto object-contain"
            />
          </a>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            {navLinks.map((link) => {
              const isDemo = link.to === '/dashboard';
              return (
                <a
                  key={link.to}
                  href={link.to}
                  onClick={(e) => handleNavClick(link.to, e)}
                  className={
                    isDemo
                      ? 'px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm font-semibold shadow-md cursor-pointer'
                      : `px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
                          isActive(link.to)
                            ? 'text-primary-600 border-b-2 border-primary-600'
                            : 'text-gray-700 hover:text-primary-600'
                        }`
                  }
                >
                  {link.label}
                </a>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-2">
            {navLinks.map((link) => {
              const isDemo = link.to === '/dashboard';
              return (
                <a
                  key={link.to}
                  href={link.to}
                  onClick={(e) => handleNavClick(link.to, e)}
                  className={
                    isDemo
                      ? 'block px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-semibold text-center shadow-md cursor-pointer'
                      : `block px-3 py-2 rounded-lg text-sm font-medium cursor-pointer ${
                          isActive(link.to)
                            ? 'bg-primary-50 text-primary-600'
                            : 'text-gray-700 hover:bg-gray-50'
                        }`
                  }
                >
                  {link.label}
                </a>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
}

