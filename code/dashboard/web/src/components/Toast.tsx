'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { X, CheckCircle, AlertTriangle, Info, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  showToast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((message: string, type: ToastType = 'info') => {
    const id = Date.now().toString();
    setToasts((prev) => [...prev, { id, message, type }]);

    // Auto-remove after 4 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const getIcon = (type: ToastType) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-accent-success" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-accent-danger" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-accent-warning" />;
      default:
        return <Info className="w-5 h-5 text-accent-info" />;
    }
  };

  const getStyles = (type: ToastType) => {
    switch (type) {
      case 'success':
        return 'border-accent-success/30 bg-accent-success/10';
      case 'error':
        return 'border-accent-danger/30 bg-accent-danger/10';
      case 'warning':
        return 'border-accent-warning/30 bg-accent-warning/10';
      default:
        return 'border-accent-info/30 bg-accent-info/10';
    }
  };

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      
      {/* Toast container */}
      <div className="fixed bottom-4 right-4 z-10000 flex flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={cn(
              'flex items-center gap-3 px-4 py-3 rounded-lg border backdrop-blur-xs shadow-lg animate-slide-in',
              'min-w-[300px] max-w-[400px]',
              getStyles(toast.type)
            )}
          >
            {getIcon(toast.type)}
            <span className="flex-1 text-white">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="p-1 hover:bg-white/10 rounded-sm"
            >
              <X className="w-4 h-4 text-white/50" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}


