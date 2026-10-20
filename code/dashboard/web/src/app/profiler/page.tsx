'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { getCompileAnalysis, getNcuSummary, getProfilePairs } from '@/lib/api';
import { cn, formatMs, formatNumber } from '@/lib/utils';
import type {
  CompileAnalysis,
  NcuKernelRow,
  NcuSummaryResult,
  ProfilePair,
  ProfilePairsResult,
} from '@/types';
import { RefreshCw, Search, Activity, Code2 } from 'lucide-react';

function joinPath(dir: string, file: string) {
  if (!dir) return file;
  if (dir.endsWith('/')) return dir + file;
  return `${dir}/${file}`;
}

function fmtMaybeNumber(v: number | null | undefined, suffix?: string) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const rendered = typeof v === 'number' ? formatNumber(v, 2) : String(v);
  return suffix ? `${rendered}${suffix}` : rendered;
}

function metric(row: NcuKernelRow, key: string): number | null {
  const v = row.metrics?.[key];
  if (typeof v !== 'number' || Number.isNaN(v)) return null;
  return v;
}

function formatSignedMs(ms: number): string {
  const sign = ms < 0 ? '-' : '';
  return sign + formatMs(Math.abs(ms));
}

function timeMs(row: NcuKernelRow): number {
  const sum = row.time_sum_ms;
  const avg = row.time_avg_ms;
  if (typeof sum === 'number' && !Number.isNaN(sum)) return sum;
  if (typeof avg === 'number' && !Number.isNaN(avg)) return avg;
  return 0;
}

type KernelDiffRow = {
  kernel_name: string;
  baseline_ms: number;
  optimized_ms: number;
  delta_ms: number;
  ratio: number | null;
};

export default function ProfilerPage() {
  const { showToast } = useToast();

  const [pairsResult, setPairsResult] = useState<ProfilePairsResult | null>(null);
  const [loadingPairs, setLoadingPairs] = useState(true);
  const [pairsError, setPairsError] = useState<string | null>(null);
  const [pairsSearch, setPairsSearch] = useState('');
  const [selectedPair, setSelectedPair] = useState<ProfilePair | null>(null);

  const [compile, setCompile] = useState<CompileAnalysis | null>(null);
  const [loadingCompile, setLoadingCompile] = useState(true);

  const [baselineFile, setBaselineFile] = useState<string>('');
  const [optimizedFile, setOptimizedFile] = useState<string>('');
  const [topK, setTopK] = useState<number>(15);
  const [timeoutSeconds, setTimeoutSeconds] = useState<number>(60);
  const [loadingNcu, setLoadingNcu] = useState(false);
  const [baselineSummary, setBaselineSummary] = useState<NcuSummaryResult | null>(null);
  const [optimizedSummary, setOptimizedSummary] = useState<NcuSummaryResult | null>(null);

  const loadPairs = useCallback(async () => {
    try {
      setLoadingPairs(true);
      setPairsError(null);
      const data = (await getProfilePairs()) as ProfilePairsResult;
      setPairsResult(data);
      if (!selectedPair && data.pairs?.length) {
        setSelectedPair(data.pairs[0]);
      }
    } catch (e) {
      setPairsError(e instanceof Error ? e.message : 'Failed to load profile pairs');
      setPairsResult(null);
    } finally {
      setLoadingPairs(false);
    }
  }, [selectedPair]);

  const loadCompile = useCallback(async () => {
    try {
      setLoadingCompile(true);
      const data = (await getCompileAnalysis()) as CompileAnalysis;
      setCompile(data);
    } catch {
      setCompile(null);
    } finally {
      setLoadingCompile(false);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadPairs(), loadCompile()]);
  }, [loadCompile, loadPairs]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!selectedPair) return;
    // Pick first available baseline/optimized NCU reports by default.
    const base = selectedPair.baseline_ncu?.[0] || '';
    const opt = selectedPair.optimized_ncu?.[0] || '';
    setBaselineFile(base);
    setOptimizedFile(opt);
    setBaselineSummary(null);
    setOptimizedSummary(null);
  }, [selectedPair]);

  const filteredPairs = useMemo(() => {
    const pairs = pairsResult?.pairs || [];
    const needle = pairsSearch.trim().toLowerCase();
    if (!needle) return pairs;
    return pairs.filter((p) => {
      const hay = `${p.chapter} ${p.run_id} ${p.name} ${p.path}`.toLowerCase();
      return hay.includes(needle);
    });
  }, [pairsResult?.pairs, pairsSearch]);

  const baselinePath = selectedPair && baselineFile ? joinPath(selectedPair.path, baselineFile) : '';
  const optimizedPath = selectedPair && optimizedFile ? joinPath(selectedPair.path, optimizedFile) : '';

  const runNcuSummary = useCallback(async () => {
    if (!baselinePath || !optimizedPath) {
      showToast('Select both a baseline and optimized NCU report.', 'warning');
      return;
    }
    try {
      setLoadingNcu(true);
      const [base, opt] = (await Promise.all([
        getNcuSummary({ report_path: baselinePath, top_k: topK, timeout_seconds: timeoutSeconds }),
        getNcuSummary({ report_path: optimizedPath, top_k: topK, timeout_seconds: timeoutSeconds }),
      ])) as [NcuSummaryResult, NcuSummaryResult];
      setBaselineSummary(base);
      setOptimizedSummary(opt);
      if (base.success && opt.success) {
        showToast('NCU summaries loaded.', 'success');
      } else {
        showToast(base.error || opt.error || 'NCU summary failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'NCU summary failed.', 'error');
      setBaselineSummary(null);
      setOptimizedSummary(null);
    } finally {
      setLoadingNcu(false);
    }
  }, [baselinePath, optimizedPath, showToast, timeoutSeconds, topK]);

  const diffRows = useMemo((): KernelDiffRow[] => {
    const baseKernels = baselineSummary?.success ? baselineSummary.kernels || [] : [];
    const optKernels = optimizedSummary?.success ? optimizedSummary.kernels || [] : [];
    if (!baseKernels.length && !optKernels.length) return [];

    const baseMap = new Map<string, number>();
    const optMap = new Map<string, number>();

    baseKernels.forEach((k) => {
      const t = timeMs(k);
      const prev = baseMap.get(k.kernel_name);
      if (prev === undefined || t > prev) baseMap.set(k.kernel_name, t);
    });
    optKernels.forEach((k) => {
      const t = timeMs(k);
      const prev = optMap.get(k.kernel_name);
      if (prev === undefined || t > prev) optMap.set(k.kernel_name, t);
    });

    const names = new Set<string>(Array.from(baseMap.keys()).concat(Array.from(optMap.keys())));
    const rows: KernelDiffRow[] = [];
    names.forEach((name) => {
      const b = baseMap.get(name) ?? 0;
      const o = optMap.get(name) ?? 0;
      const ratio = o > 0 ? b / o : null;
      rows.push({
        kernel_name: name,
        baseline_ms: b,
        optimized_ms: o,
        delta_ms: b - o,
        ratio,
      });
    });
    return rows.sort((a, b) => b.baseline_ms - a.baseline_ms).slice(0, 25);
  }, [baselineSummary, optimizedSummary]);

  const selectedName = selectedPair?.name || '—';

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Top kernels, Nsight artifacts, and compile analysis."
      actions={
        <button
          onClick={refreshAll}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
          disabled={loadingPairs || loadingCompile}
        >
          <RefreshCw className={cn('w-4 h-4', loadingPairs || loadingCompile ? 'animate-spin' : '')} />
          Refresh
        </button>
      }
    >
      <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="card xl:col-span-1">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-accent-warning" />
              <h2 className="text-lg font-semibold text-white">Profile Pairs</h2>
            </div>
            <span className="text-xs text-white/50">
              {pairsResult?.count ?? 0} found
            </span>
          </div>
          <div className="px-5 py-3 border-b border-white/5">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
              <input
                type="text"
                placeholder="Search pairs..."
                value={pairsSearch}
                onChange={(e) => setPairsSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-hidden focus:border-accent-primary/50 text-sm"
              />
            </div>
          </div>
          <div className="card-body p-0">
            {loadingPairs ? (
              <div className="p-5 text-white/60 text-sm">Loading profile pairs...</div>
            ) : pairsError ? (
              <div className="p-5 text-white/70 text-sm">{pairsError}</div>
            ) : filteredPairs.length === 0 ? (
              <div className="p-5 text-white/60 text-sm">
                No profile pairs found. Run <span className="font-mono">aisp bench run --profile deep_dive</span>.
              </div>
            ) : (
              <div className="max-h-[620px] overflow-y-auto hide-scrollbar">
                {filteredPairs.map((pair) => {
                  const isActive = selectedPair?.path === pair.path;
                  return (
                    <button
                      key={pair.path}
                      onClick={() => setSelectedPair(pair)}
                      className={cn(
                        'w-full text-left px-4 py-3 border-b border-white/5 hover:bg-white/5 transition-colors',
                        isActive && 'bg-accent-primary/10'
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-medium text-white truncate">
                          {pair.chapter || pair.name}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={cn('badge', pair.has_ncu ? 'badge-success' : 'badge-info')}>
                            NCU {pair.has_ncu ? 'yes' : 'no'}
                          </span>
                          <span className={cn('badge', pair.has_nsys ? 'badge-success' : 'badge-info')}>
                            NSYS {pair.has_nsys ? 'yes' : 'no'}
                          </span>
                        </div>
                      </div>
                      <div className="text-xs text-white/50 mt-1 truncate font-mono">{pair.run_id}</div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="xl:col-span-2 space-y-6">
          <div className="card">
            <div className="card-header">
              <div>
                <h2 className="text-lg font-semibold text-white">NCU Top Kernels</h2>
                <p className="text-xs text-white/50">From an existing <span className="font-mono">.ncu-rep</span> (or exported raw CSV).</p>
              </div>
              <span className="text-xs text-white/50 truncate max-w-[420px] font-mono">{selectedName}</span>
            </div>
            <div className="card-body space-y-4">
              {!selectedPair ? (
                <div className="text-white/60 text-sm">Select a profile pair to inspect kernels.</div>
              ) : !selectedPair.has_ncu ? (
                <div className="text-white/60 text-sm">This pair has no NCU reports.</div>
              ) : (
                <>
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                    <label className="space-y-1 lg:col-span-2">
                      <div className="text-xs uppercase text-white/40">Baseline report</div>
                      <select
                        value={baselineFile}
                        onChange={(e) => setBaselineFile(e.target.value)}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                      >
                        {selectedPair.baseline_ncu.map((f) => (
                          <option key={`base-${f}`} value={f}>
                            {f}
                          </option>
                        ))}
                      </select>
                      <div className="text-[11px] text-white/40 font-mono truncate">{baselinePath}</div>
                    </label>

                    <label className="space-y-1 lg:col-span-2">
                      <div className="text-xs uppercase text-white/40">Optimized report</div>
                      <select
                        value={optimizedFile}
                        onChange={(e) => setOptimizedFile(e.target.value)}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                      >
                        {selectedPair.optimized_ncu.map((f) => (
                          <option key={`opt-${f}`} value={f}>
                            {f}
                          </option>
                        ))}
                      </select>
                      <div className="text-[11px] text-white/40 font-mono truncate">{optimizedPath}</div>
                    </label>

                    <label className="space-y-1">
                      <div className="text-xs uppercase text-white/40">top_k</div>
                      <input
                        type="number"
                        min={1}
                        max={200}
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                      />
                    </label>

                    <label className="space-y-1">
                      <div className="text-xs uppercase text-white/40">timeout_s</div>
                      <input
                        type="number"
                        min={1}
                        max={3600}
                        value={timeoutSeconds}
                        onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                      />
                    </label>

                    <div className="lg:col-span-2 flex items-end">
                      <button
                        onClick={runNcuSummary}
                        className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
                        disabled={loadingNcu}
                      >
                        <RefreshCw className={cn('w-4 h-4', loadingNcu ? 'animate-spin' : '')} />
                        {loadingNcu ? 'Loading...' : 'Load Top Kernels'}
                      </button>
                    </div>
                  </div>

                  {(baselineSummary?.error || optimizedSummary?.error) && (
                    <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                      {baselineSummary?.error || optimizedSummary?.error}
                    </div>
                  )}

                  {baselineSummary?.success && optimizedSummary?.success && diffRows.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-xs uppercase text-white/40">Baseline vs Optimized (top kernels by baseline time)</div>
                      <div className="overflow-x-auto rounded-lg border border-white/10">
                        <table className="min-w-[900px] w-full text-sm">
                          <thead className="bg-white/5 text-white/60">
                            <tr>
                              <th className="text-left px-3 py-2">Kernel</th>
                              <th className="text-right px-3 py-2">Baseline</th>
                              <th className="text-right px-3 py-2">Optimized</th>
                              <th className="text-right px-3 py-2">Delta</th>
                              <th className="text-right px-3 py-2">Ratio</th>
                            </tr>
                          </thead>
                          <tbody>
                            {diffRows.map((row) => (
                              <tr key={`diff-${row.kernel_name}`} className="border-t border-white/5">
                                <td className="px-3 py-2 text-white/90 font-mono text-xs max-w-[520px] truncate">
                                  {row.kernel_name}
                                </td>
                                <td className="px-3 py-2 text-right text-white/80">{formatMs(row.baseline_ms)}</td>
                                <td className="px-3 py-2 text-right text-white/80">{formatMs(row.optimized_ms)}</td>
                                <td className={cn('px-3 py-2 text-right', row.delta_ms >= 0 ? 'text-accent-success' : 'text-accent-danger')}>
                                  {formatSignedMs(row.delta_ms)}
                                </td>
                                <td className="px-3 py-2 text-right text-white/70">
                                  {row.ratio ? `${row.ratio.toFixed(2)}x` : '—'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-white">Baseline</div>
                        <span className={cn('badge', baselineSummary?.success ? 'badge-success' : 'badge-danger')}>
                          {baselineSummary?.success ? 'OK' : 'FAILED'}
                        </span>
                      </div>
                      {baselineSummary?.success && baselineSummary.kernels && (
                        <div className="overflow-x-auto rounded-lg border border-white/10">
                          <table className="min-w-[900px] w-full text-sm">
                            <thead className="bg-white/5 text-white/60">
                              <tr>
                                <th className="text-left px-3 py-2">Kernel</th>
                                <th className="text-right px-3 py-2">Time</th>
                                <th className="text-right px-3 py-2">Time %</th>
                                <th className="text-right px-3 py-2">SM</th>
                                <th className="text-right px-3 py-2">DRAM</th>
                                <th className="text-right px-3 py-2">L2</th>
                                <th className="text-right px-3 py-2">Occ</th>
                                <th className="text-right px-3 py-2">Regs</th>
                                <th className="text-right px-3 py-2">Shmem</th>
                                <th className="text-left px-3 py-2">Occ limit</th>
                              </tr>
                            </thead>
                            <tbody>
                              {baselineSummary.kernels.map((k) => (
                                <tr key={`base-k-${k.id}`} className="border-t border-white/5">
                                  <td className="px-3 py-2 text-white/90 font-mono text-xs max-w-[420px] truncate">
                                    {k.kernel_name}
                                  </td>
                                  <td className="px-3 py-2 text-right text-white/80">{formatMs(timeMs(k))}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(k.time_pct ?? null, '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'lts__throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'sm__warps_active.avg.pct_of_peak_sustained_active'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'launch__registers_per_thread'))}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'launch__shared_mem_per_block'))}</td>
                                  <td className="px-3 py-2 text-white/60 font-mono text-xs">{k.occupancy_limit_reason || '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-white">Optimized</div>
                        <span className={cn('badge', optimizedSummary?.success ? 'badge-success' : 'badge-danger')}>
                          {optimizedSummary?.success ? 'OK' : 'FAILED'}
                        </span>
                      </div>
                      {optimizedSummary?.success && optimizedSummary.kernels && (
                        <div className="overflow-x-auto rounded-lg border border-white/10">
                          <table className="min-w-[900px] w-full text-sm">
                            <thead className="bg-white/5 text-white/60">
                              <tr>
                                <th className="text-left px-3 py-2">Kernel</th>
                                <th className="text-right px-3 py-2">Time</th>
                                <th className="text-right px-3 py-2">Time %</th>
                                <th className="text-right px-3 py-2">SM</th>
                                <th className="text-right px-3 py-2">DRAM</th>
                                <th className="text-right px-3 py-2">L2</th>
                                <th className="text-right px-3 py-2">Occ</th>
                                <th className="text-right px-3 py-2">Regs</th>
                                <th className="text-right px-3 py-2">Shmem</th>
                                <th className="text-left px-3 py-2">Occ limit</th>
                              </tr>
                            </thead>
                            <tbody>
                              {optimizedSummary.kernels.map((k) => (
                                <tr key={`opt-k-${k.id}`} className="border-t border-white/5">
                                  <td className="px-3 py-2 text-white/90 font-mono text-xs max-w-[420px] truncate">
                                    {k.kernel_name}
                                  </td>
                                  <td className="px-3 py-2 text-right text-white/80">{formatMs(timeMs(k))}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(k.time_pct ?? null, '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'lts__throughput.avg.pct_of_peak_sustained_elapsed'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'sm__warps_active.avg.pct_of_peak_sustained_active'), '%')}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'launch__registers_per_thread'))}</td>
                                  <td className="px-3 py-2 text-right text-white/70">{fmtMaybeNumber(metric(k, 'launch__shared_mem_per_block'))}</td>
                                  <td className="px-3 py-2 text-white/60 font-mono text-xs">{k.occupancy_limit_reason || '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-accent-primary" />
                <h2 className="text-lg font-semibold text-white">torch.compile Analysis</h2>
              </div>
              {loadingCompile ? (
                <span className="text-xs text-white/50">Loading…</span>
              ) : (
                <span className={cn('badge', compile?.has_real_data ? 'badge-success' : 'badge-info')}>
                  {compile?.has_real_data ? 'Real data' : 'Heuristic'}
                </span>
              )}
            </div>
            <div className="card-body space-y-4">
              {!compile ? (
                <div className="text-white/60 text-sm">No compile analysis available.</div>
              ) : (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase text-white/40">Avg speedup</div>
                      <div className="text-2xl font-semibold text-white mt-1">
                        {compile.speedup ? `${compile.speedup.toFixed(2)}x` : '—'}
                      </div>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase text-white/40">Compile time</div>
                      <div className="text-2xl font-semibold text-white mt-1">
                        {compile.compile_time_ms ? formatMs(compile.compile_time_ms) : '—'}
                      </div>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase text-white/40">Graph breaks</div>
                      <div className="text-2xl font-semibold text-white mt-1">
                        {compile.graph_breaks ?? 0}
                      </div>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase text-white/40">Fusion ratio</div>
                      <div className="text-2xl font-semibold text-white mt-1">
                        {compile.fusion_ratio ? compile.fusion_ratio.toFixed(2) : '—'}
                      </div>
                    </div>
                  </div>

                  {Array.isArray(compile.recommendations) && compile.recommendations.length > 0 && (
                    <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase text-white/40 mb-2">Recommendations</div>
                      <ul className="space-y-1 text-sm text-white/80">
                        {compile.recommendations.map((rec, idx) => (
                          <li key={`rec-${idx}`} className="flex items-start gap-2">
                            <span className="mt-2 h-1.5 w-1.5 rounded-full bg-accent-primary" />
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {compile.compile_benchmarks?.length ? (
                    <div className="overflow-x-auto rounded-lg border border-white/10">
                      <table className="min-w-[800px] w-full text-sm">
                        <thead className="bg-white/5 text-white/60">
                          <tr>
                            <th className="text-left px-3 py-2">Benchmark</th>
                            <th className="text-left px-3 py-2">Chapter</th>
                            <th className="text-right px-3 py-2">Speedup</th>
                            <th className="text-right px-3 py-2">Baseline</th>
                            <th className="text-right px-3 py-2">Optimized</th>
                          </tr>
                        </thead>
                        <tbody>
                          {compile.compile_benchmarks.map((b) => (
                            <tr key={`${b.chapter}-${b.name}`} className="border-t border-white/5">
                              <td className="px-3 py-2 text-white/90 font-mono text-xs max-w-[420px] truncate">{b.name}</td>
                              <td className="px-3 py-2 text-white/70">{b.chapter}</td>
                              <td className="px-3 py-2 text-right text-white/80">{b.speedup?.toFixed(2)}x</td>
                              <td className="px-3 py-2 text-right text-white/70">{formatMs(b.baseline_time_ms)}</td>
                              <td className="px-3 py-2 text-right text-white/70">{formatMs(b.optimized_time_ms)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-white/60 text-sm">
                      No compile-focused benchmarks detected in recent results.
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </section>
    </DashboardShell>
  );
}
