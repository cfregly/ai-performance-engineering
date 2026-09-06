'use client';

import { useEffect, useRef, useState } from 'react';
import { GpuInfo } from '@/types';

interface UseGpuStreamOptions {
  enabled?: boolean;
  intervalMs?: number;
}

interface GpuStreamState {
  gpu: GpuInfo | null;
  status: 'idle' | 'connected' | 'error';
  lastUpdated: string | null;
}

export function useGpuStream(options: UseGpuStreamOptions = {}): GpuStreamState {
  const { enabled = true, intervalMs = 5000 } = options;
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [status, setStatus] = useState<'idle' | 'connected' | 'error'>('idle');
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    const interval = Math.max(1000, intervalMs);
    const url = `/api/gpu/stream?interval=${Math.round(interval / 1000)}`;
    const source = new EventSource(url);
    sourceRef.current = source;

    const markStreamError = () => {
      setGpu((current) => (current ? { ...current, live: false } : current));
      setStatus('error');
    };

    source.addEventListener('gpu', (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data || '{}');
        if (payload.gpu) {
          setGpu({ ...payload.gpu, live: true });
          setLastUpdated(payload.timestamp || new Date().toISOString());
          setStatus('connected');
        }
      } catch {
        markStreamError();
      }
    });

    source.onerror = markStreamError;

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [enabled, intervalMs]);

  return { gpu, status, lastUpdated };
}
