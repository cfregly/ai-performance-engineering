'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  getBenchmarkHistory,
  getBenchmarkOverview,
  getGpuInfo,
} from '@/lib/api';
import { useGpuStream } from '@/lib/useGpuStream';
import { DashboardShell } from '@/components/DashboardShell';
import { StatsCard } from '@/components/StatsCard';
import { SpeedupChart } from '@/components/SpeedupChart';
import { StatusChart } from '@/components/StatusChart';
import { BenchmarkTable } from '@/components/BenchmarkTable';
import { GpuCard } from '@/components/GpuCard';
import { SoftwareStackWidget } from '@/components/SoftwareStackWidget';
import { DependenciesWidget } from '@/components/DependenciesWidget';
import { AIAssistantTab } from '@/components/tabs/AIAssistantTab';
import { Zap, TrendingUp, CheckCircle, Clock, AlertTriangle, RefreshCw } from 'lucide-react';
import type { BenchmarkHistory, BenchmarkOverview, BenchmarkSummary } from '@/types';

const emptySummary: BenchmarkSummary = {
  total: 0,
  succeeded: 0,
  failed: 0,
  skipped: 0,
  avg_speedup: 0,
  max_speedup: 0,
  min_speedup: 0,
};

type ChartFilter = { type: 'benchmark' | 'chapter'; value: string } | null;

function buildTrend(current: number, previous?: number | null) {
  if (previous === undefined || previous === null || previous === 0) return undefined;
  return {
    value: ((current - previous) / previous) * 100,
    label: 'vs last run',
  };
}

function OverviewSkeleton() {
  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="card p-6 animate-pulse">
          <div className="h-5 w-40 bg-white/10 rounded mb-4" />
          <div className="space-y-3">
            <div className="h-2 bg-white/10 rounded" />
            <div className="h-2 bg-white/10 rounded w-5/6" />
            <div className="h-2 bg-white/10 rounded w-2/3" />
          </div>
        </div>
        <div className="xl:col-span-2 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, index) => (
              <div key={`stat-skel-${index}`} className="card p-5 animate-pulse">
                <div className="h-3 w-24 bg-white/10 rounded mb-3" />
                <div className="h-8 w-32 bg-white/10 rounded" />
                <div className="h-3 w-20 bg-white/10 rounded mt-3" />
              </div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Array.from({ length: 2 }).map((_, index) => (
              <div key={`chart-skel-${index}`} className="card p-6 animate-pulse">
                <div className="h-4 w-32 bg-white/10 rounded mb-4" />
                <div className="h-40 bg-white/5 rounded" />
              </div>
            ))}
          </div>
        </div>
      </section>
      <div className="card p-6 animate-pulse">
        <div className="h-4 w-32 bg-white/10 rounded mb-4" />
        <div className="h-32 bg-white/5 rounded" />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [overview, setOverview] = useState<BenchmarkOverview | null>(null);
  const [history, setHistory] = useState<BenchmarkHistory | null>(null);
  const [gpuInfo, setGpuInfo] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [chartFilter, setChartFilter] = useState<ChartFilter>(null);

  const { gpu: streamGpu, lastUpdated: gpuLastUpdated } = useGpuStream({
    enabled: true,
    intervalMs: 5000,
  });

  useEffect(() => {
    if (streamGpu) {
      setGpuInfo(streamGpu);
    }
  }, [streamGpu]);

  const loadOverview = useCallback(async (isRefresh = false) => {
    try {
      if (!isRefresh) {
        setLoading(true);
      }
      setError(null);
      const [overviewData, historyData, gpu] = await Promise.all([
        getBenchmarkOverview(),
        getBenchmarkHistory(),
        getGpuInfo(),
      ]);
      setOverview(overviewData as BenchmarkOverview);
      setHistory(historyData as BenchmarkHistory);
      setGpuInfo(gpu);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to connect to backend');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  const handleRefresh = () => {
    setRefreshing(true);
    setRefreshKey((value) => value + 1);
    loadOverview(true);
  };

  const summary = overview?.summary || emptySummary;
  const statusCounts = overview?.status_counts || { succeeded: 0, failed: 0, skipped: 0 };
  const topSpeedups = overview?.top_speedups || [];
  const chapterStats = overview?.chapter_stats || [];

  const chapters = useMemo(
    () => (overview?.chapter_stats || []).map((entry) => entry.chapter).sort(),
    [overview?.chapter_stats]
  );

  const previousRun = history?.runs?.[1];
  const currentSuccessRate = summary.total ? (summary.succeeded / summary.total) * 100 : 0;
  const previousSuccessRate = previousRun && previousRun.benchmark_count
    ? (previousRun.successful / previousRun.benchmark_count) * 100
    : null;

  const lastUpdated = (() => {
    if (!overview?.timestamp) return '—';
    try {
      return new Date(overview.timestamp).toLocaleString();
    } catch {
      return overview.timestamp;
    }
  })();

  const handleChartSelect = (selection: { type: 'benchmark' | 'chapter'; name: string }) => {
    setChartFilter({ type: selection.type, value: selection.name });
    const table = document.getElementById('benchmark-table');
    if (table) {
      table.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle={`MCP-aligned telemetry • Last updated ${lastUpdated}`}
      onRefresh={handleRefresh}
      searchTargetId="benchmark-search"
      actions={
        <button
          onClick={handleRefresh}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white"
          disabled={refreshing}
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      }
      headerWidgets={
        <>
          <SoftwareStackWidget />
          <DependenciesWidget />
        </>
      }
    >
      {loading && !overview ? (
        <OverviewSkeleton />
      ) : error && !overview ? (
        <div className="card">
          <div className="card-body text-center py-16">
            <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
            <p className="text-white/70 mb-4">{error}</p>
            <button
              onClick={handleRefresh}
              className="px-6 py-2 bg-accent-primary text-black rounded-lg font-medium hover:opacity-90"
            >
              Retry Connection
            </button>
          </div>
        </div>
      ) : (
        <>
          <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {gpuInfo && (
              <div className="xl:col-span-1">
                <GpuCard gpu={gpuInfo} lastUpdated={gpuLastUpdated} />
              </div>
            )}
            <div className="xl:col-span-2 space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <StatsCard
                  title="Average Speedup"
                  value={`${summary.avg_speedup?.toFixed(2) || 0}x`}
                  subtitle="Across all benchmarks"
                  icon={Zap}
                  variant="success"
                  trend={buildTrend(summary.avg_speedup, previousRun?.avg_speedup)}
                />
                <StatsCard
                  title="Max Speedup"
                  value={`${summary.max_speedup?.toFixed(2) || 0}x`}
                  subtitle="Best optimization"
                  icon={TrendingUp}
                  variant="default"
                  trend={buildTrend(summary.max_speedup, previousRun?.max_speedup)}
                />
                <StatsCard
                  title="Success Rate"
                  value={`${summary.total ? ((summary.succeeded / summary.total) * 100).toFixed(0) : 0}%`}
                  subtitle={`${summary.succeeded}/${summary.total || 0} passed`}
                  icon={CheckCircle}
                  variant={summary.succeeded === summary.total ? 'success' : 'warning'}
                  trend={buildTrend(currentSuccessRate, previousSuccessRate)}
                />
                <StatsCard
                  title="Total Benchmarks"
                  value={summary.total || 0}
                  subtitle={lastUpdated}
                  icon={Clock}
                  trend={buildTrend(summary.total, previousRun?.benchmark_count)}
                />
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SpeedupChart
                  benchmarks={topSpeedups}
                  chapterStats={chapterStats}
                  onSelect={handleChartSelect}
                />
                <StatusChart counts={statusCounts} />
              </div>
            </div>
          </section>

          <section>
            <BenchmarkTable
              chapters={chapters}
              externalFilter={chartFilter || undefined}
              onClearExternalFilter={() => setChartFilter(null)}
              refreshKey={refreshKey}
            />
          </section>

          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
              <p className="text-sm text-white/50">Run MCP tools directly from the UI.</p>
            </div>
            <AIAssistantTab />
          </section>
        </>
      )}
    </DashboardShell>
  );
}
