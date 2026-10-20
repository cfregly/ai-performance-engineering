'use client';

import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: number;
    label: string;
  };
  variant?: 'default' | 'success' | 'warning' | 'danger';
  className?: string;
}

const variantStyles = {
  default: 'from-accent-primary to-accent-secondary',
  success: 'from-accent-success to-accent-primary',
  warning: 'from-accent-warning to-accent-tertiary',
  danger: 'from-accent-danger to-accent-tertiary',
};

export function StatsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  variant = 'default',
  className,
}: StatsCardProps) {
  return (
    <div className={cn('card p-5', className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-white/50 mb-1">{title}</p>
          <p className={cn(
            'text-3xl font-bold bg-linear-to-r bg-clip-text text-transparent',
            variantStyles[variant]
          )}>
            {value}
          </p>
          {subtitle && (
            <p className="text-xs text-white/40 mt-1">{subtitle}</p>
          )}
          {trend && (
            <div className={cn(
              'inline-flex items-center gap-1 mt-2 px-2 py-0.5 rounded-full text-xs',
              trend.value >= 0 ? 'bg-accent-success/20 text-accent-success' : 'bg-accent-danger/20 text-accent-danger'
            )}>
              <span>{trend.value >= 0 ? '↑' : '↓'}</span>
              <span>{Math.abs(trend.value).toFixed(1)}%</span>
              <span className="text-white/40">{trend.label}</span>
            </div>
          )}
        </div>
        {Icon && (
          <div className={cn(
            'p-3 rounded-xl bg-linear-to-br',
            variantStyles[variant],
            'bg-opacity-20'
          )}>
            <Icon className="w-6 h-6 text-white/80" />
          </div>
        )}
      </div>
    </div>
  );
}


