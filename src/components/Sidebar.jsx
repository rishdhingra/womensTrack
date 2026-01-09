import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Upload, FileText, Users, FileCheck } from 'lucide-react';

export default function Sidebar() {
  const location = useLocation();
  
  const menuItems = [
    { to: '/dashboard', icon: LayoutDashboard, label: 'Overview' },
    { to: '/dashboard?tab=upload', icon: Upload, label: 'Upload' },
    { to: '/dashboard?tab=outputs', icon: FileText, label: 'Outputs' },
    { to: '/dashboard?tab=cohorts', icon: Users, label: 'Cohorts' },
    { to: '/dashboard?tab=reports', icon: FileCheck, label: 'Reports' },
  ];

  const isActive = (path) => {
    if (path === '/dashboard') {
      return location.pathname === '/dashboard' && !location.search;
    }
    return location.pathname === '/dashboard' && location.search.includes(path.split('?')[1]);
  };

  return (
    <aside className="w-64 bg-white border-r border-gray-200 min-h-screen p-4">
      <div className="mb-6">
        <h2 className="text-lg font-serif font-bold text-gray-900">Dashboard</h2>
        <p className="text-xs text-gray-500 mt-1">Research Use Only</p>
      </div>
      
      <nav className="space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.to);
          
          return (
            <Link
              key={item.to}
              to={item.to}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                active
                  ? 'bg-primary-50 text-primary-700 font-semibold'
                  : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Icon size={20} />
              <span className="text-sm">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

