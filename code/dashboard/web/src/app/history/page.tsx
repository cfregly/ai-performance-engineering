'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { DashboardShell } from '@/components/DashboardShell';
import { StatsCard } from '@/components/StatsCard';
import { getBenchmarkHistory, getBenchmarkTrends } from '@/lib/api';
import type { BenchmarkHistory, BenchmarkRunSummary, BenchmarkTrends } from '@/types';
import { TrendingUp, Clock, Database, Zap, ArrowLeftRight, Sparkles } from 'lucide-react';

function HistorySkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={`history-skel-${index}`} className="card p-5 animate-pulse">
            <div className="h-3 w-24 bg-white/10 rounded-sm mb-3" />
            <div className="h-8 w-32 bg-white/10 rounded-sm" />
            <div className="h-3 w-20 bg-white/10 rounded-sm mt-3" />
          </div>
        ))}
      </div>
      <div className="card p-6 animate-pulse">
        <div className="h-4 w-40 bg-white/10 rounded-sm mb-4" />
        <div className="h-48 bg-white/5 rounded-sm" />
      </div>
    </div>
  );
}

export default function HistoryPage() {
  const [history, setHistory] = useState<BenchmarkHistory | null>(null);
  const [trends, setTrends] = useState<BenchmarkTrends | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<BenchmarkRunSummary | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [historyData, trendsData] = await Promise.all([
        getBenchmarkHistory(),
        getBenchmarkTrends(),
      ]);
      setHistory(historyData as BenchmarkHistory);
      setTrends(trendsData as BenchmarkTrends);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load benchmark history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const runCount = history?.total_runs || 0;
  const latestRun = history?.runs?.[0];
  const bestEver = trends?.best_ever?.speedup || 0;
  const avgSpeedup = trends?.avg_speedup || 0;

  const chartData = useMemo(() => trends?.history || [], [trends?.history]);
  const improvements = trends?.improvements || [];
  const regressions = trends?.regressions || [];
  const recentImprovement = improvements[improvements.length - 1];
  const recentRegression = regressions[regressions.length - 1];

  useEffect(() => {
    if (!selectedRun && latestRun) {
      setSelectedRun(latestRun);
    }
  }, [latestRun, selectedRun]);

  const buildCompareUrl = (baseline: string, candidate: string) =>
    `/compare?baseline=${encodeURIComponent(baseline)}&candidate=${encodeURIComponent(candidate)}`;

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Benchmark history and long-term performance trends."
    >
      {loading ? (
        <HistorySkeleton />
      ) : error ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">
            {error}
          </div>
        </div>
      ) : runCount === 0 ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">
            No benchmark runs found yet.
          </div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            <StatsCard
              title="Total Runs"
              value={runCount}
              subtitle="Recorded benchmark sessions"
              icon={Database}
            />
            <StatsCard
              title="Average Speedup"
              value={`${avgSpeedup.toFixed(2)}x`}
              subtitle="Across all runs"
              icon={TrendingUp}
            />
            <StatsCard
              title="Best Ever"
              value={`${bestEver.toFixed(2)}x`}
              subtitle={trends?.best_ever?.date || '-'}
              icon={Zap}
              variant="success"
            />
            <StatsCard
              title="Latest Run"
              value={latestRun?.date || '-'}
              subtitle={`${latestRun?.benchmark_count || 0} benchmarks`}
              icon={Clock}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card lg:col-span-2">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Speedup Trend</h2>
                <span className="badge badge-info">{chartData.length} points</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="date" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
                    <YAxis tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(16, 16, 24, 0.95)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number, name) => [
                        `${value.toFixed(2)}x`,
                        name === 'avg_speedup' ? 'Avg speedup' : 'Max speedup',
                      ]}
                    />
                    <Line
                      type="monotone"
                      dataKey="avg_speedup"
                      stroke="#00f5d4"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="max_speedup"
                      stroke="#f72585"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Recent Movement</h2>
                <Sparkles className="w-4 h-4 text-accent-secondary" />
              </div>
              <div className="card-body space-y-3 text-sm text-white/70">
                <div className="flex items-center justify-between">
                  <span>Latest improvement</span>
                  <span className="text-accent-success">
                    {recentImprovement ? `+${recentImprovement.delta.toFixed(2)}x` : '-'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Latest regression</span>
                  <span className="text-accent-danger">
                    {recentRegression ? recentRegression.delta.toFixed(2) + 'x' : '-'}
                  </span>
                </div>
                {latestRun && selectedRun && latestRun.source !== selectedRun.source && (
                  <Link
                    href={buildCompareUrl(selectedRun.source, latestRun.source)}
                    className="mt-3 inline-flex items-center gap-2 text-xs text-accent-primary hover:text-accent-secondary"
                  >
                    <ArrowLeftRight className="w-3 h-3" />
                    Compare selected run to latest
                  </Link>
                )}
              </div>
            </div>
          </div>

          {selectedRun && latestRun && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Selected Run</h2>
                {selectedRun.source !== latestRun.source && (
                  <Link
                    href={buildCompareUrl(selectedRun.source, latestRun.source)}
                    className="inline-flex items-center gap-2 text-sm text-accent-primary hover:text-accent-secondary"
                  >
                    <ArrowLeftRight className="w-4 h-4" />
                    Compare to latest
                  </Link>
                )}
              </div>
              <div className="card-body grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-white/70">
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Run Date</div>
                  <div className="text-white">{selectedRun.date}</div>
                  <div className="text-xs text-white/40 mt-1">{selectedRun.source}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Speedup</div>
                  <div className="text-white">
                    Avg {selectedRun.avg_speedup.toFixed(2)}x | Max {selectedRun.max_speedup.toFixed(2)}x
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Success</div>
                  <div className="text-white">
                    {selectedRun.successful}/{selectedRun.benchmark_count} succeeded
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-white">Run History</h2>
              <span className="badge badge-info">{runCount} runs</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5 text-xs uppercase text-white/50">
                    <th className="px-5 py-3 text-left">Date</th>
                    <th className="px-5 py-3 text-right">Benchmarks</th>
                    <th className="px-5 py-3 text-right">Avg Speedup</th>
                    <th className="px-5 py-3 text-right">Max Speedup</th>
                    <th className="px-5 py-3 text-right">Success</th>
                    <th className="px-5 py-3 text-right">Compare</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {history?.runs?.map((run) => (
                    <tr
                      key={run.timestamp}
                      className={`text-sm text-white/70 hover:bg-white/2 transition-colors ${selectedRun?.source === run.source ? 'bg-white/5' : ''}`}
                      onClick={() => setSelectedRun(run)}
                    >
                      <td className="px-5 py-3">{run.date}</td>
                      <td className="px-5 py-3 text-right">{run.benchmark_count}</td>
                      <td className="px-5 py-3 text-right">{run.avg_speedup.toFixed(2)}x</td>
                      <td className="px-5 py-3 text-right">{run.max_speedup.toFixed(2)}x</td>
                      <td className="px-5 py-3 text-right">
                        {run.successful}/{run.benchmark_count}
                      </td>
                      <td className="px-5 py-3 text-right">
                        {latestRun && run.source !== latestRun.source ? (
                          <Link
                            href={buildCompareUrl(run.source, latestRun.source)}
                            className="text-xs text-accent-primary hover:text-accent-secondary"
                            onClick={(event) => event.stopPropagation()}
                          >
                            Compare
                          </Link>
                        ) : (
                          <span className="text-xs text-white/30">Latest</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </DashboardShell>
  );
}
