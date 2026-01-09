export default function Card({ children, className = '', hover = true }) {
  const baseClasses = 'glass-card p-6';
  const hoverClasses = hover ? 'hover:shadow-xl' : '';
  
  return (
    <div className={`${baseClasses} ${hoverClasses} ${className}`}>
      {children}
    </div>
  );
}

