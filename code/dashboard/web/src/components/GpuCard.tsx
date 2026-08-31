'use client';

import { GpuInfo } from '@/types';
import { formatBytes, mebibytesToBytes } from '@/lib/utils';
import { Cpu, Thermometer, Zap, HardDrive } from 'lucide-react';

interface GpuCardProps {
  gpu: GpuInfo;
  lastUpdated?: string | null;
}

export function GpuCard({ gpu, lastUpdated }: GpuCardProps) {
  const memoryBytesTotal = typeof gpu.memory_total === 'number' ? mebibytesToBytes(gpu.memory_total) : 0;
  const memoryBytesUsed = typeof gpu.memory_used === 'number' ? mebibytesToBytes(gpu.memory_used) : 0;
  const memoryPercent =
    memoryBytesTotal > 0 ? Math.min(100, (memoryBytesUsed / memoryBytesTotal) * 100) : 0;
  const powerDraw = gpu.power_draw ?? gpu.power;
  const powerLimit = gpu.power_limit;
  const powerPercent =
    powerDraw && powerLimit ? Math.min(100, (powerDraw / powerLimit) * 100) : null;
  const utilization = gpu.utilization ?? 0;
  let liveStamp: string | null = null;
  if (lastUpdated) {
    try {
      liveStamp = new Date(lastUpdated).toLocaleTimeString();
    } catch {
      liveStamp = null;
    }
  }

  return (
    <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-accent-secondary to-accent-tertiary">
                <Cpu className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white">{gpu.name}</h3>
                <p className="text-xs text-white/50">
                  CUDA Runtime {gpu.cuda_version || '—'} | Driver {gpu.driver_version || '—'} | CC {gpu.compute_capability || '—'}
                </p>
              </div>
            </div>
            {gpu.live && (
              <div className="flex items-center gap-2 text-xs text-accent-success">
                <span className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
                <span>Live{liveStamp ? ` • ${liveStamp}` : ''}</span>
              </div>
            )}
          </div>
      <div className="card-body space-y-4">
        {/* Memory */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-white/70">
              <HardDrive className="w-4 h-4" />
              <span>Memory</span>
            </div>
            <span className="text-sm font-medium text-white">
              {formatBytes(memoryBytesUsed)} / {formatBytes(memoryBytesTotal)}
            </span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full transition-all duration-500"
              style={{ width: `${memoryPercent}%` }}
            />
          </div>
        </div>

        {/* Utilization */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-white/70">
              <Cpu className="w-4 h-4" />
              <span>Utilization</span>
            </div>
            <span className="text-sm font-medium text-white">{utilization}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent-success to-accent-primary rounded-full transition-all duration-500"
              style={{ width: `${utilization}%` }}
            />
          </div>
        </div>

        {/* Temperature & Power */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-white/5">
          <div className="flex items-center gap-2">
            <Thermometer className="w-4 h-4 text-accent-warning" />
            <div>
              <p className="text-lg font-bold text-white">{gpu.temperature}°C</p>
              <p className="text-xs text-white/50">Temperature</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-accent-info" />
            <div>
              <p className="text-lg font-bold text-white">{powerDraw ? `${powerDraw}W` : '—'}</p>
              <p className="text-xs text-white/50">
                Power{powerPercent !== null && powerLimit ? ` (${powerPercent.toFixed(0)}% of ${powerLimit}W)` : ''}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
