'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Navigation, DASHBOARD_TABS } from '@/components/Navigation';

interface DashboardShellProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  headerWidgets?: React.ReactNode;
  onRefresh?: () => void;
  searchTargetId?: string;
  children: React.ReactNode;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}

export function DashboardShell({
  title,
  subtitle,
  actions,
  headerWidgets,
  onRefresh,
  searchTargetId,
  children,
}: DashboardShellProps) {
  const router = useRouter();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }
      if (event.key === 'r' && onRefresh) {
        event.preventDefault();
        onRefresh();
        return;
      }
      if (event.key === '/' && searchTargetId) {
        event.preventDefault();
        const el = document.getElementById(searchTargetId) as HTMLInputElement | null;
        if (el) {
          el.focus();
          el.select();
        }
        return;
      }
      const matchedTab = DASHBOARD_TABS.find((tab) => tab.shortcut === event.key);
      if (matchedTab) {
        event.preventDefault();
        router.push(matchedTab.href);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onRefresh, router, searchTargetId]);

  return (
    <div className="min-h-screen">
      <header className="px-4 lg:px-8 py-6 border-b border-white/5 bg-brand-bg/70 backdrop-blur-xs">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-white">{title}</h1>
            {subtitle && <p className="text-sm text-white/50">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-3">{actions}</div>}
        </div>
        <div className="mt-4">
          <Navigation />
        </div>
        {headerWidgets && (
          <div className="mt-4 flex flex-wrap items-center gap-3">{headerWidgets}</div>
        )}
      </header>
      <main className="px-4 lg:px-8 pb-10 pt-8 space-y-8">{children}</main>
    </div>
  );
}
