'use client';

import { useCallback, useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { runClockLockCheck } from '@/lib/api';
import { cn, formatNumber } from '@/lib/utils';
import type { ClockLockCheckResult, ClockLockResultRow } from '@/types';
import { RefreshCw, ShieldCheck, ShieldX } from 'lucide-react';

function badge(ok: boolean | undefined) {
  if (ok) return 'badge badge-success';
  return 'badge badge-danger';
}

function fmt(v: number | string | null | undefined) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return String(v);
  return String(v);
}

function fmtFloat(v: number | null | undefined) {
  if (v === null || v === undefined) return '—';
  return formatNumber(v, 2);
}

function rowClass(row: ClockLockResultRow) {
  if (row.locked) return 'border-accent-success/20';
  return 'border-accent-danger/20';
}

export default function SystemPage() {
  const { showToast } = useToast();
  const [smClock, setSmClock] = useState<string>('');
  const [memClock, setMemClock] = useState<string>('');
  const [devices, setDevices] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ClockLockCheckResult | null>(null);

  const runCheck = useCallback(async () => {
    try {
      setRunning(true);
      const params: Record<string, unknown> = {};
      if (smClock.trim()) params.sm_clock_mhz = Number(smClock);
      if (memClock.trim()) params.mem_clock_mhz = Number(memClock);
      if (devices.trim()) params.devices = devices;
      const data = (await runClockLockCheck(params)) as ClockLockCheckResult;
      setResult(data);
      if (data.success) {
        showToast('Clock locking validated successfully.', 'success');
      } else {
        showToast(data.error || 'Clock locking validation failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Clock locking validation failed.', 'error');
      setResult(null);
    } finally {
      setRunning(false);
    }
  }, [devices, memClock, showToast, smClock]);

  const rows = result?.results || [];

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="System checks that gate benchmark validity."
      actions={
        <button
          onClick={runCheck}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
          disabled={running}
        >
          <RefreshCw className={cn('w-4 h-4', running ? 'animate-spin' : '')} />
          Run Check
        </button>
      }
    >
      <div className="grid grid-cols-1 gap-6">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                {result?.success ? (
                  <ShieldCheck className="w-5 h-5 text-accent-success" />
                ) : (
                  <ShieldX className="w-5 h-5 text-accent-warning" />
                )}
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Clock Lock Check</h2>
                <p className="text-xs text-white/50">
                  Validates harness-based GPU clock locking. Canonical benchmarks must be locked.
                </p>
              </div>
            </div>
            {result && (
              <span className={badge(result.success)}>
                {result.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">SM Clock (MHz)</div>
                <input
                  value={smClock}
                  onChange={(e) => setSmClock(e.target.value)}
                  placeholder="(optional) e.g. 1500"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">MEM Clock (MHz)</div>
                <input
                  value={memClock}
                  onChange={(e) => setMemClock(e.target.value)}
                  placeholder="(optional) leave blank for max supported"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">Devices</div>
                <input
                  value={devices}
                  onChange={(e) => setDevices(e.target.value)}
                  placeholder="(optional) comma list e.g. 0,1,2,3"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
            </div>

            {result?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {result.error}
              </div>
            )}

            {rows.length > 0 && (
              <div className="space-y-3">
                <div className="text-xs text-white/50">
                  GPUs checked: {result?.gpu_count ?? '—'}
                </div>
                <div className="overflow-x-auto rounded-lg border border-white/10">
                  <table className="min-w-[1100px] w-full text-sm">
                    <thead className="bg-white/5 text-white/60">
                      <tr>
                        <th className="text-left px-3 py-2">Device</th>
                        <th className="text-left px-3 py-2">Physical</th>
                        <th className="text-left px-3 py-2">Locked</th>
                        <th className="text-left px-3 py-2">App SM (B/D/A)</th>
                        <th className="text-left px-3 py-2">App MEM (B/D/A)</th>
                        <th className="text-left px-3 py-2">Cur SM (B/D/A)</th>
                        <th className="text-left px-3 py-2">Cur MEM (B/D/A)</th>
                        <th className="text-left px-3 py-2">Peak FP16 TFLOPS</th>
                        <th className="text-left px-3 py-2">Peak GB/s</th>
                        <th className="text-left px-3 py-2">Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr
                          key={`clk-${row.device}-${row.physical_index}`}
                          className={cn('border-t border-white/5', rowClass(row))}
                        >
                          <td className="px-3 py-2 text-white">{row.device}</td>
                          <td className="px-3 py-2 text-white/80">{row.physical_index}</td>
                          <td className="px-3 py-2">
                            <span className={badge(row.locked)}>{row.locked ? 'Yes' : 'No'}</span>
                          </td>
                          <td className="px-3 py-2 text-white/80 font-mono text-xs">
                            {fmt(row.before?.app_sm_mhz)}/{fmt(row.during?.app_sm_mhz)}/{fmt(row.after?.app_sm_mhz)}
                          </td>
                          <td className="px-3 py-2 text-white/80 font-mono text-xs">
                            {fmt(row.before?.app_mem_mhz)}/{fmt(row.during?.app_mem_mhz)}/{fmt(row.after?.app_mem_mhz)}
                          </td>
                          <td className="px-3 py-2 text-white/80 font-mono text-xs">
                            {fmt(row.before?.cur_sm_mhz)}/{fmt(row.during?.cur_sm_mhz)}/{fmt(row.after?.cur_sm_mhz)}
                          </td>
                          <td className="px-3 py-2 text-white/80 font-mono text-xs">
                            {fmt(row.before?.cur_mem_mhz)}/{fmt(row.during?.cur_mem_mhz)}/{fmt(row.after?.cur_mem_mhz)}
                          </td>
                          <td className="px-3 py-2 text-white/70">{fmtFloat(row.theoretical_tflops_fp16 ?? null)}</td>
                          <td className="px-3 py-2 text-white/70">{fmtFloat(row.theoretical_gbps ?? null)}</td>
                          <td className="px-3 py-2 text-white/70">{row.error || row.physical_index_error || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardShell>
  );
}

