import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, decimals = 2): string {
  if (n >= 1e9) return (n / 1e9).toFixed(decimals) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(decimals) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(decimals) + 'K';
  return n.toFixed(decimals);
}

export function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(2)}μs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
  return bytes + ' B';
}

export function mebibytesToBytes(mebibytes: number): number {
  return mebibytes * 1024 ** 2;
}

export function getSpeedupColor(speedup: number): string {
  if (speedup >= 5) return '#00f5a0';   // Excellent
  if (speedup >= 2) return '#00f5d4';   // Great
  if (speedup >= 1.5) return '#4cc9f0'; // Good
  if (speedup >= 1) return '#ffc43d';   // Marginal
  return '#ff4757';                      // Regression
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'succeeded': return '#00f5a0';
    case 'failed': return '#ff4757';
    case 'skipped': return '#ffc43d';
    default: return '#6b7280';
  }
}

