'use client';

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Activity, Clock3, RefreshCw, Route, ShieldCheck, TerminalSquare } from 'lucide-react';
import { DashboardShell } from '@/components/DashboardShell';
import { StatsCard } from '@/components/StatsCard';
import { getBenchmarkE2eStatus } from '@/lib/api';
import type { BenchmarkE2EIssueGroup, BenchmarkE2EStatusSnapshot, BenchmarkE2EStatusStage } from '@/types';

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return `${value.toFixed(2)}%`;
}

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  if (value < 60) {
    return `${value.toFixed(0)}s`;
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatEventLabel(event: Record<string, unknown>): string {
  const topLevel = String(event.event || '').trim();
  if (topLevel) return topLevel;
  const child = String(event.event_type || '').trim();
  if (child) return child;
  return 'event';
}

function statusVariant(value: string | null | undefined): 'default' | 'success' | 'warning' | 'danger' {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed' || normalized === 'succeeded' || normalized === 'success') {
    return 'success';
  }
  if (normalized.includes('aborted') || normalized === 'failed' || normalized === 'error') {
    return 'danger';
  }
  if (normalized.includes('stale') || normalized === 'partial' || normalized === 'warning') {
    return 'warning';
  }
  return 'default';
}

function E2EStatusSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={`e2e-skel-${index}`} className="card p-5 animate-pulse">
            <div className="h-3 w-24 bg-white/10 rounded-sm mb-3" />
            <div className="h-8 w-32 bg-white/10 rounded-sm" />
            <div className="h-3 w-20 bg-white/10 rounded-sm mt-3" />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Array.from({ length: 2 }).map((_, index) => (
          <div key={`e2e-card-${index}`} className="card p-6 animate-pulse">
            <div className="h-4 w-36 bg-white/10 rounded-sm mb-4" />
            <div className="space-y-3">
              <div className="h-3 bg-white/10 rounded-sm" />
              <div className="h-3 bg-white/10 rounded-sm w-5/6" />
              <div className="h-3 bg-white/10 rounded-sm w-2/3" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StageRow({ stage }: { stage: BenchmarkE2EStatusStage }) {
  const attemptCount = stage.attempts?.length || 0;
  const issueCount = stage.issues?.length || 0;
  const failedBenchmarks = stage.failed_benchmarks?.length || 0;

  return (
    <tr className="border-b border-white/5 text-sm text-white/75">
      <td className="px-5 py-3 align-top text-white">{stage.name}</td>
      <td className="px-5 py-3 align-top">{stage.status}</td>
      <td className="px-5 py-3 align-top">{attemptCount}</td>
      <td className="px-5 py-3 align-top">{failedBenchmarks}</td>
      <td className="px-5 py-3 align-top">{issueCount}</td>
      <td className="px-5 py-3 align-top text-white/55">{stage.description || '-'}</td>
    </tr>
  );
}

function IssueGroupRow({ group }: { group: BenchmarkE2EIssueGroup }) {
  return (
    <tr className="border-b border-white/5 text-sm text-white/75">
      <td className="px-5 py-3 align-top text-white">{group.stage}</td>
      <td className="px-5 py-3 align-top">{group.count}</td>
      <td className="px-5 py-3 align-top">{group.signature}</td>
      <td className="px-5 py-3 align-top text-white/60">{group.sample_targets?.join(', ') || '-'}</td>
      <td className="px-5 py-3 align-top text-white/55">{group.root_cause_hint || '-'}</td>
    </tr>
  );
}

function E2EPageContent() {
  const searchParams = useSearchParams();
  const runId = searchParams?.get('run_id') || undefined;

  const [status, setStatus] = useState<BenchmarkE2EStatusSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadStatus = useCallback(async (isRefresh = false) => {
    try {
      if (!isRefresh) {
        setLoading(true);
      }
      setError(null);
      const payload = await getBenchmarkE2eStatus({ run_id: runId, recent_events: 10 });
      setStatus(payload as BenchmarkE2EStatusSnapshot);
      if ((payload as BenchmarkE2EStatusSnapshot).success === false) {
        setError((payload as BenchmarkE2EStatusSnapshot).error || 'Failed to load e2e status');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load e2e status');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [runId]);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      loadStatus(true);
    }, 15000);
    return () => window.clearInterval(timer);
  }, [loadStatus]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadStatus(true);
  };

  const stageRows = useMemo(() => status?.stages || [], [status?.stages]);
  const issueGroups = useMemo(() => status?.issue_groups || [], [status?.issue_groups]);
  const recentEvents = useMemo(() => status?.recent_events || [], [status?.recent_events]);
  const notes = status?.notes || [];
  const progressSource = status?.progress_source;
  const current = status?.current;
  const watcher = status?.watcher;
  const actions = status?.actions;
  const ledgerSummary = status?.ledgers?.summary;
  const reportedFailures = status?.current?.reported_failures || [];
  const latestTimestamp = progressSource?.progress_timestamp || '—';
  const activeIssueCount = ledgerSummary?.active_issue_count ?? ledgerSummary?.issue_count ?? 0;
  const activeUnresolvedCount = ledgerSummary?.active_unresolved_count ?? ledgerSummary?.unresolved_count ?? 0;
  const historicalIssueCount = ledgerSummary?.historical_issue_count ?? 0;
  const issueSummary = ledgerSummary
    ? `${activeUnresolvedCount} active / ${historicalIssueCount} historical`
    : '-';

  return (
    <DashboardShell
      title="E2E Sweep"
      subtitle={`Normalized live status for ${status?.run_id || runId || 'the latest e2e run'}`}
      onRefresh={handleRefresh}
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
    >
      {loading && !status ? (
        <E2EStatusSkeleton />
      ) : error && (!status || status.success === false) ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">{error}</div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            <StatsCard
              title="Run State"
              value={status?.inferred_state || '-'}
              subtitle={`overall=${status?.overall_status || '-'}`}
              icon={Activity}
              variant={statusVariant(status?.inferred_state)}
            />
            <StatsCard
              title="Current Progress"
              value={formatPercent(current?.percent_complete)}
              subtitle={current?.step || 'idle'}
              icon={Route}
            />
            <StatsCard
              title="Incidents"
              value={activeIssueCount}
              subtitle={issueSummary}
              icon={ShieldCheck}
              variant={activeUnresolvedCount > 0 ? 'danger' : historicalIssueCount > 0 ? 'warning' : 'success'}
            />
            <StatsCard
              title="Watcher"
              value={watcher?.watch_state || 'not armed'}
              subtitle={`auto-resume=${watcher?.auto_resume_count ?? 0}${
                watcher?.watcher_live === false ? ' | dead' : ''
              }${
                watcher?.stored_watch_state && watcher.stored_watch_state !== watcher.watch_state
                  ? ` | stored=${watcher.stored_watch_state}`
                  : ''
              }`}
              icon={ShieldCheck}
              variant={watcher?.watch_state === 'watching' ? 'success' : 'warning'}
            />
            <StatsCard title="Current Timestamp" value={String(latestTimestamp)} subtitle={formatSeconds(current?.elapsed_seconds)} icon={Clock3} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Source of Truth</h2>
                <span className="badge badge-info">{progressSource?.kind || 'unknown'}</span>
              </div>
              <div className="card-body space-y-4 text-sm text-white/70">
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Reason</div>
                  <div>{progressSource?.reason || 'No source metadata available.'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Progress Path</div>
                  <div className="font-mono break-all text-xs text-white/60">
                    {progressSource?.progress_path || '-'}
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Summary Lag</div>
                    <div>{formatSeconds(progressSource?.summary_progress_lag_seconds)}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Checkpoint Lag</div>
                    <div>{formatSeconds(progressSource?.checkpoint_progress_lag_seconds)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Effective Overall</div>
                    <div>{status?.overall_status || '-'}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Stored Overall</div>
                    <div>{status?.stored_overall_status || status?.overall_status || '-'}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Effective Resume</div>
                    <div>{String(status?.resume_available ?? '-')}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-1">Stored Resume</div>
                    <div>{String(status?.stored_resume_available ?? status?.resume_available ?? '-')}</div>
                  </div>
                </div>
                {notes.length > 0 && (
                  <div>
                    <div className="text-xs uppercase text-white/40 mb-2">Notes</div>
                    <div className="space-y-2">
                      {notes.map((note) => (
                        <div key={note} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/65">
                          {note}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Current Execution</h2>
                <TerminalSquare className="w-4 h-4 text-accent-secondary" />
              </div>
              <div className="card-body grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-white/70">
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Stage</div>
                  <div className="text-white">{current?.stage || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Bucket</div>
                  <div className="text-white">{current?.bucket || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Detail</div>
                  <div>{current?.detail || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">ETA</div>
                  <div>{formatSeconds(current?.eta_seconds)}</div>
                </div>
                <div className="sm:col-span-2">
                  <div className="text-xs uppercase text-white/40 mb-1">Child Run</div>
                  <div className="font-mono break-all text-xs text-white/60">{current?.child_run_id || '-'}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Actions</h2>
                <span className="badge badge-info">CLI / API / MCP</span>
              </div>
              <div className="card-body space-y-4 text-sm text-white/70">
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Preferred CLI</div>
                  <div className="font-mono break-all text-xs text-white/60">{actions?.status_command_shell || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Resume CLI</div>
                  <div className="font-mono break-all text-xs text-white/60">{actions?.resume_command_shell || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">API Path</div>
                  <div className="font-mono break-all text-xs text-white/60">{actions?.status_api_path || '-'}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Preferred MCP Tool</div>
                  <div>{actions?.preferred_mcp_tool || '-'}</div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Recent Events</h2>
                <span className="badge badge-info">{recentEvents.length}</span>
              </div>
              <div className="card-body space-y-3 text-xs text-white/65">
                {recentEvents.length === 0 ? (
                  <div>No recent events recorded.</div>
                ) : (
                  recentEvents.map((event, index) => (
                    <div key={`${formatEventLabel(event)}-${index}`} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-white">{formatEventLabel(event)}</span>
                        <span className="text-white/40">{String(event.ts || event.timestamp || '-')}</span>
                      </div>
                      <div className="mt-1 text-white/55">
                        {String(event.stage || event.status || event.error || event.run_id || '').trim() || 'no extra detail'}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {reportedFailures.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Reported Failures</h2>
                <span className="badge badge-warning">{reportedFailures.length}</span>
              </div>
              <div className="card-body space-y-3 text-sm text-white/70">
                {reportedFailures.map((entry, index) => (
                  <div key={`${String(entry.target || 'failure')}-${index}`} className="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-white">{String(entry.target || '-')}</div>
                      <div className="text-xs uppercase text-white/40">{String(entry.status || '-')}</div>
                    </div>
                    <div className="mt-1 text-xs text-white/55">
                      {String(entry.error || entry.symptom || entry.source || 'No extra detail')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {issueGroups.length > 0 && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Grouped Incidents</h2>
                <span className="badge badge-warning">{issueGroups.length} groups</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="text-left text-xs uppercase tracking-wide text-white/40">
                    <tr>
                      <th className="px-5 py-3">Stage</th>
                      <th className="px-5 py-3">Count</th>
                      <th className="px-5 py-3">Signature</th>
                      <th className="px-5 py-3">Sample Targets</th>
                      <th className="px-5 py-3">Hint</th>
                    </tr>
                  </thead>
                  <tbody>
                    {issueGroups.map((group) => (
                      <IssueGroupRow key={group.group_id} group={group} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-white">Stage Ledger</h2>
              <span className="badge badge-info">{stageRows.length} stages</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5 text-xs uppercase text-white/50">
                    <th className="px-5 py-3 text-left">Stage</th>
                    <th className="px-5 py-3 text-left">Status</th>
                    <th className="px-5 py-3 text-left">Attempts</th>
                    <th className="px-5 py-3 text-left">Failed Benchmarks</th>
                    <th className="px-5 py-3 text-left">Issues</th>
                    <th className="px-5 py-3 text-left">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {stageRows.map((stage) => (
                    <StageRow key={stage.name} stage={stage} />
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

export default function E2EPage() {
  return (
    <Suspense fallback={<E2EStatusSkeleton />}>
      <E2EPageContent />
    </Suspense>
  );
}
