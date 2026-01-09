export default function Section({ children, className = '', id }) {
  const isFullScreen = className.includes('min-h-screen') && className.includes('flex');
  return (
    <section id={id} className={`py-16 md:py-24 ${className}`}>
      <div className={`container mx-auto px-4 md:px-6 ${isFullScreen ? 'h-full flex flex-col justify-center' : ''}`}>
        {children}
      </div>
    </section>
  );
}

