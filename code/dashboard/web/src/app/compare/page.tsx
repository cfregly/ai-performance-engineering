'use client';

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { DashboardShell } from '@/components/DashboardShell';
import { StatsCard } from '@/components/StatsCard';
import { getBenchmarkCompare, getBenchmarkHistory } from '@/lib/api';
import { cn, formatMs, getSpeedupColor } from '@/lib/utils';
import type {
  BenchmarkCompareDelta,
  BenchmarkCompareResult,
  BenchmarkHistory,
  BenchmarkRunSummary,
} from '@/types';
import {
  ArrowLeftRight,
  AlertTriangle,
  RefreshCw,
  Search,
  TrendingDown,
  TrendingUp,
  Filter,
  Layers,
} from 'lucide-react';

function formatRunLabel(run: BenchmarkRunSummary) {
  return `${run.date} | ${run.benchmark_count} benchmarks`;
}

function formatDelta(value: number) {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}x`;
}

function statusBadge(status: string) {
  if (status === 'succeeded') return 'bg-accent-success/20 text-accent-success';
  if (status === 'failed') return 'bg-accent-danger/20 text-accent-danger';
  if (status === 'skipped') return 'bg-white/10 text-white/50';
  return 'bg-white/10 text-white/50';
}

function ComparePageInner() {
  const searchParams = useSearchParams();
  const [history, setHistory] = useState<BenchmarkHistory | null>(null);
  const [compare, setCompare] = useState<BenchmarkCompareResult | null>(null);
  const [baselinePath, setBaselinePath] = useState<string>('');
  const [candidatePath, setCandidatePath] = useState<string>('');
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [filterMode, setFilterMode] = useState<'all' | 'improvements' | 'regressions' | 'status-changed'>('all');
  const [sortMode, setSortMode] = useState<'delta_desc' | 'delta_asc' | 'name_asc'>('delta_desc');
  const initRef = useRef(false);

  const loadHistory = useCallback(async () => {
    try {
      setLoadingHistory(true);
      const data = await getBenchmarkHistory();
      setHistory(data as BenchmarkHistory);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load benchmark history');
    } finally {
      setLoadingHistory(false);
    }
  }, []);

  const decodeParam = (value: string | null) => {
    if (!value) return null;
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  };

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!history || initRef.current) return;
    const runs = history.runs || [];
    const queryBaseline = decodeParam(searchParams?.get('baseline') ?? null);
    const queryCandidate = decodeParam(searchParams?.get('candidate') ?? null);

    const defaultCandidate = queryCandidate || runs[0]?.source || '';
    const defaultBaseline = queryBaseline || runs[1]?.source || runs[0]?.source || '';

    setBaselinePath(defaultBaseline);
    setCandidatePath(defaultCandidate);
    initRef.current = true;
  }, [history, searchParams]);

  const runByPath = useMemo(() => {
    const map = new Map<string, BenchmarkRunSummary>();
    history?.runs?.forEach((run) => {
      map.set(run.source, run);
    });
    return map;
  }, [history]);

  const loadCompare = useCallback(async () => {
    if (!baselinePath || !candidatePath) return;
    if (baselinePath === candidatePath) {
      setCompare(null);
      setError('Select two different runs to compare.');
      return;
    }
    try {
      setLoadingCompare(true);
      setError(null);
      const data = await getBenchmarkCompare({
        baseline: baselinePath,
        candidate: candidatePath,
        top: 8,
      });
      setCompare(data as BenchmarkCompareResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to compare runs');
      setCompare(null);
    } finally {
      setLoadingCompare(false);
    }
  }, [baselinePath, candidatePath]);

  useEffect(() => {
    loadCompare();
  }, [loadCompare]);

  const swapRuns = () => {
    setBaselinePath(candidatePath);
    setCandidatePath(baselinePath);
  };

  const filteredDeltas = useMemo(() => {
    if (!compare) return [] as BenchmarkCompareDelta[];
    const needle = search.toLowerCase();
    let rows = compare.deltas;
    if (needle) {
      rows = rows.filter(
        (d) =>
          d.name.toLowerCase().includes(needle) ||
          d.chapter.toLowerCase().includes(needle) ||
          d.key.toLowerCase().includes(needle)
      );
    }
    if (filterMode === 'improvements') {
      rows = rows.filter((d) => d.delta > 0);
    }
    if (filterMode === 'regressions') {
      rows = rows.filter((d) => d.delta < 0);
    }
    if (filterMode === 'status-changed') {
      rows = rows.filter((d) => d.status_changed);
    }
    if (sortMode === 'delta_desc') {
      rows = [...rows].sort((a, b) => b.delta - a.delta);
    }
    if (sortMode === 'delta_asc') {
      rows = [...rows].sort((a, b) => a.delta - b.delta);
    }
    if (sortMode === 'name_asc') {
      rows = [...rows].sort((a, b) => a.key.localeCompare(b.key));
    }
    return rows;
  }, [compare, search, filterMode, sortMode]);

  const baselineRun = runByPath.get(baselinePath);
  const candidateRun = runByPath.get(candidatePath);
  const summaryBaseline = compare?.baseline.summary;
  const summaryCandidate = compare?.candidate.summary;
  const avgDelta = summaryCandidate && summaryBaseline
    ? summaryCandidate.avg_speedup - summaryBaseline.avg_speedup
    : 0;
  const successRateBase = summaryBaseline && summaryBaseline.total
    ? (summaryBaseline.succeeded / summaryBaseline.total) * 100
    : 0;
  const successRateCandidate = summaryCandidate && summaryCandidate.total
    ? (summaryCandidate.succeeded / summaryCandidate.total) * 100
    : 0;
  const successRateDelta = successRateCandidate - successRateBase;

  const transitionEntries = compare
    ? Object.entries(compare.status_transitions).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <DashboardShell title="AI Performance Dashboard" subtitle="Compare benchmark runs side-by-side.">
      {loadingHistory ? (
        <div className="card">
          <div className="card-body text-white/60">Loading history...</div>
        </div>
      ) : !history || (history.runs?.length || 0) < 2 ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">
            <AlertTriangle className="w-10 h-10 text-accent-warning mx-auto mb-4" />
            <p>At least two benchmark runs are required to compare.</p>
            <p className="text-xs text-white/40 mt-2">Run benchmarks to generate more history.</p>
          </div>
        </div>
      ) : (
        <>
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card lg:col-span-2">
              <div className="card-header">
                <div>
                  <h2 className="text-lg font-semibold text-white">Run Selection</h2>
                  <p className="text-xs text-white/50">Pick two runs to compare performance deltas.</p>
                </div>
                <button
                  onClick={swapRuns}
                  className="flex items-center gap-2 px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white/70 hover:text-white"
                >
                  <ArrowLeftRight className="w-4 h-4" />
                  Swap
                </button>
              </div>
              <div className="card-body grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="text-xs uppercase text-white/40">Baseline</div>
                  <select
                    value={baselinePath}
                    onChange={(e) => setBaselinePath(e.target.value)}
                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                  >
                    {history.runs.map((run) => (
                      <option key={run.source} value={run.source}>
                        {formatRunLabel(run)}
                      </option>
                    ))}
                  </select>
                  {baselineRun && (
                    <div className="text-xs text-white/50">
                      Avg {baselineRun.avg_speedup.toFixed(2)}x | Max {baselineRun.max_speedup.toFixed(2)}x
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  <div className="text-xs uppercase text-white/40">Candidate</div>
                  <select
                    value={candidatePath}
                    onChange={(e) => setCandidatePath(e.target.value)}
                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                  >
                    {history.runs.map((run) => (
                      <option key={run.source} value={run.source}>
                        {formatRunLabel(run)}
                      </option>
                    ))}
                  </select>
                  {candidateRun && (
                    <div className="text-xs text-white/50">
                      Avg {candidateRun.avg_speedup.toFixed(2)}x | Max {candidateRun.max_speedup.toFixed(2)}x
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Change Summary</h2>
                <button
                  onClick={loadCompare}
                  className="flex items-center gap-2 px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white/70 hover:text-white"
                >
                  <RefreshCw className={`w-4 h-4 ${loadingCompare ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
              <div className="card-body space-y-3 text-sm text-white/70">
                <div className="flex items-center justify-between">
                  <span>Overlap</span>
                  <span className="text-white">{compare?.overlap.common ?? 0}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Added</span>
                  <span className="text-white">{compare?.overlap.added ?? 0}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Removed</span>
                  <span className="text-white">{compare?.overlap.removed ?? 0}</span>
                </div>
                {transitionEntries.length > 0 && (
                  <div className="pt-3 border-t border-white/5">
                    <div className="text-xs uppercase text-white/40 mb-2">Status changes</div>
                    <div className="space-y-1">
                      {transitionEntries.slice(0, 3).map(([transition, count]) => (
                        <div key={transition} className="flex items-center justify-between text-xs">
                          <span>{transition}</span>
                          <span className="text-white">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>

          {error && (
            <div className="card">
              <div className="card-body text-center text-accent-danger">{error}</div>
            </div>
          )}

          {compare && (
            <>
              <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <StatsCard
                  title="Baseline Avg"
                  value={`${summaryBaseline?.avg_speedup.toFixed(2) || '0.00'}x`}
                  subtitle={`${summaryBaseline?.succeeded || 0}/${summaryBaseline?.total || 0} succeeded`}
                  icon={Layers}
                />
                <StatsCard
                  title="Candidate Avg"
                  value={`${summaryCandidate?.avg_speedup.toFixed(2) || '0.00'}x`}
                  subtitle={`${summaryCandidate?.succeeded || 0}/${summaryCandidate?.total || 0} succeeded`}
                  icon={TrendingUp}
                  variant={avgDelta >= 0 ? 'success' : 'danger'}
                />
                <StatsCard
                  title="Avg Delta"
                  value={formatDelta(avgDelta)}
                  subtitle={avgDelta >= 0 ? 'Net improvement' : 'Net regression'}
                  icon={avgDelta >= 0 ? TrendingUp : TrendingDown}
                  variant={avgDelta >= 0 ? 'success' : 'danger'}
                />
                <StatsCard
                  title="Success Rate"
                  value={`${successRateCandidate.toFixed(0)}%`}
                  subtitle={`Delta ${successRateDelta >= 0 ? '+' : ''}${successRateDelta.toFixed(1)}%`}
                  icon={successRateDelta >= 0 ? TrendingUp : TrendingDown}
                  variant={successRateDelta >= 0 ? 'success' : 'warning'}
                />
              </section>

              <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center gap-2">
                      <TrendingDown className="w-4 h-4 text-accent-danger" />
                      <h3 className="text-lg font-semibold text-white">Top Regressions</h3>
                    </div>
                    <span className="badge badge-danger">{compare.regressions.length}</span>
                  </div>
                  <div className="card-body space-y-3">
                    {compare.regressions.length === 0 ? (
                      <div className="text-sm text-white/50">No regressions detected.</div>
                    ) : (
                      compare.regressions.map((item) => (
                        <div key={item.key} className="flex items-center justify-between text-sm">
                          <div>
                            <div className="text-white">{item.chapter}:{item.name}</div>
                            <div className="text-xs text-white/40">
                              {item.baseline_speedup.toFixed(2)}x {'->'} {item.candidate_speedup.toFixed(2)}x
                            </div>
                          </div>
                          <span className="text-accent-danger font-semibold">{formatDelta(item.delta)}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-accent-success" />
                      <h3 className="text-lg font-semibold text-white">Top Improvements</h3>
                    </div>
                    <span className="badge badge-success">{compare.improvements.length}</span>
                  </div>
                  <div className="card-body space-y-3">
                    {compare.improvements.length === 0 ? (
                      <div className="text-sm text-white/50">No improvements detected.</div>
                    ) : (
                      compare.improvements.map((item) => (
                        <div key={item.key} className="flex items-center justify-between text-sm">
                          <div>
                            <div className="text-white">{item.chapter}:{item.name}</div>
                            <div className="text-xs text-white/40">
                              {item.baseline_speedup.toFixed(2)}x {'->'} {item.candidate_speedup.toFixed(2)}x
                            </div>
                          </div>
                          <span className="text-accent-success font-semibold">{formatDelta(item.delta)}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </section>

              <section className="card">
                <div className="card-header flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Benchmark Delta Table</h2>
                    <p className="text-xs text-white/50">
                      {compare.overlap.common} overlapping benchmarks - optimized timings per run
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                      <input
                        type="text"
                        placeholder="Search benchmarks..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-hidden focus:border-accent-primary/50 w-52"
                      />
                    </div>
                    <select
                      value={filterMode}
                      onChange={(e) => setFilterMode(e.target.value as typeof filterMode)}
                      className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                    >
                      <option value="all">All changes</option>
                      <option value="improvements">Improvements</option>
                      <option value="regressions">Regressions</option>
                      <option value="status-changed">Status changed</option>
                    </select>
                    <select
                      value={sortMode}
                      onChange={(e) => setSortMode(e.target.value as typeof sortMode)}
                      className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                    >
                      <option value="delta_desc">Sort by delta (desc)</option>
                      <option value="delta_asc">Sort by delta (asc)</option>
                      <option value="name_asc">Sort by name</option>
                    </select>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/5 text-xs uppercase text-white/50">
                        <th className="px-5 py-3 text-left">Benchmark</th>
                        <th className="px-5 py-3 text-right">Baseline</th>
                        <th className="px-5 py-3 text-right">Candidate</th>
                        <th className="px-5 py-3 text-right">Delta</th>
                        <th className="px-5 py-3 text-center">Status</th>
                        <th className="px-5 py-3 text-right">Opt Time (Base)</th>
                        <th className="px-5 py-3 text-right">Opt Time (Cand)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {filteredDeltas.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="px-5 py-6 text-center text-sm text-white/50">
                            No benchmarks match the current filters.
                          </td>
                        </tr>
                      ) : (
                        filteredDeltas.map((row) => (
                          <tr key={row.key} className="text-sm text-white/70">
                            <td className="px-5 py-4">
                              <div className="text-white">{row.chapter}:{row.name}</div>
                              <div className="text-xs text-white/40">
                                {row.delta_pct !== null && row.delta_pct !== undefined
                                  ? `${row.delta_pct >= 0 ? '+' : ''}${row.delta_pct.toFixed(1)}%`
                                  : '-'}
                              </div>
                            </td>
                            <td className="px-5 py-4 text-right font-mono">
                              {row.baseline_speedup.toFixed(2)}x
                            </td>
                            <td className="px-5 py-4 text-right font-mono">
                              <span style={{ color: getSpeedupColor(row.candidate_speedup) }}>
                                {row.candidate_speedup.toFixed(2)}x
                              </span>
                            </td>
                            <td className={cn(
                              'px-5 py-4 text-right font-mono font-semibold',
                              row.delta > 0 && 'text-accent-success',
                              row.delta < 0 && 'text-accent-danger'
                            )}>
                              {formatDelta(row.delta)}
                            </td>
                            <td className="px-5 py-4 text-center">
                              <span className={cn('px-2 py-1 rounded-full text-xs font-medium', statusBadge(row.candidate_status))}>
                                {row.baseline_status} {'->'} {row.candidate_status}
                              </span>
                            </td>
                            <td className="px-5 py-4 text-right font-mono text-xs">
                              {row.baseline_optimized_time_ms !== null && row.baseline_optimized_time_ms !== undefined
                                ? formatMs(row.baseline_optimized_time_ms)
                                : '-'}
                            </td>
                            <td className="px-5 py-4 text-right font-mono text-xs">
                              {row.candidate_optimized_time_ms !== null && row.candidate_optimized_time_ms !== undefined
                                ? formatMs(row.candidate_optimized_time_ms)
                                : '-'}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center gap-2">
                      <Filter className="w-4 h-4 text-accent-info" />
                      <h3 className="text-lg font-semibold text-white">Added Benchmarks</h3>
                    </div>
                    <span className="badge badge-info">{compare.added_benchmarks.length}</span>
                  </div>
                  <div className="card-body space-y-2">
                    {compare.added_benchmarks.length === 0 ? (
                      <div className="text-sm text-white/50">No new benchmarks in candidate run.</div>
                    ) : (
                      compare.added_benchmarks.map((bench) => (
                        <div key={bench.name} className="flex items-center justify-between text-sm text-white/70">
                          <span>{bench.chapter}:{bench.name}</span>
                          <span className={cn('px-2 py-1 rounded-full text-xs', statusBadge(bench.status))}>
                            {bench.status}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="flex items-center gap-2">
                      <Filter className="w-4 h-4 text-accent-warning" />
                      <h3 className="text-lg font-semibold text-white">Removed Benchmarks</h3>
                    </div>
                    <span className="badge badge-warning">{compare.removed_benchmarks.length}</span>
                  </div>
                  <div className="card-body space-y-2">
                    {compare.removed_benchmarks.length === 0 ? (
                      <div className="text-sm text-white/50">No benchmarks removed from candidate run.</div>
                    ) : (
                      compare.removed_benchmarks.map((bench) => (
                        <div key={bench.name} className="flex items-center justify-between text-sm text-white/70">
                          <span>{bench.chapter}:{bench.name}</span>
                          <span className={cn('px-2 py-1 rounded-full text-xs', statusBadge(bench.status))}>
                            {bench.status}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </section>
            </>
          )}
        </>
      )}
    </DashboardShell>
  );
}

function CompareLoading() {
  return (
    <div className="card">
      <div className="card-body text-white/60">Loading compare page...</div>
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<CompareLoading />}>
      <ComparePageInner />
    </Suspense>
  );
}
