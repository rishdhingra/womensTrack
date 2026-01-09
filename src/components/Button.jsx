import { Link } from 'react-router-dom';

export default function Button({ 
  children, 
  variant = 'primary', 
  onClick, 
  to,
  className = '',
  disabled = false,
  ...props 
}) {
  const baseClasses = 'px-6 py-3 rounded-lg font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 inline-flex items-center justify-center';
  
  const variants = {
    primary: 'bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500 shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed',
    secondary: 'bg-white text-primary-600 border-2 border-primary-600 hover:bg-primary-50 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed',
    outline: 'bg-transparent text-primary-600 border-2 border-primary-600 hover:bg-primary-50 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed',
  };

  const buttonClasses = `${baseClasses} ${variants[variant]} ${className}`;

  if (to) {
    return (
      <Link
        to={to}
        className={buttonClasses}
        {...props}
      >
        {children}
      </Link>
    );
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={buttonClasses}
      {...props}
    >
      {children}
    </button>
  );
}

