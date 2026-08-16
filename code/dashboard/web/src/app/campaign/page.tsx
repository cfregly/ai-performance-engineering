'use client';

import { FormEvent, Suspense, useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { DashboardShell } from '@/components/DashboardShell';
import { CampaignSnapshot } from '@/components/CampaignSnapshot';
import { getOptimizationCampaign } from '@/lib/api';
import type { CampaignDashboardResult } from '@/types';
import { RefreshCw } from 'lucide-react';

function CampaignPageInner() {
  const searchParams = useSearchParams();
  const [workspace, setWorkspace] = useState(searchParams?.get('workspace') || '');
  const [data, setData] = useState<CampaignDashboardResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCampaign = useCallback(async (selectedWorkspace: string) => {
    const normalized = selectedWorkspace.trim();
    if (!normalized) {
      setError('Enter a campaign workspace path.');
      setData(null);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const result = await getOptimizationCampaign(normalized);
      setData(result);
    } catch (exception) {
      setData(null);
      setError(exception instanceof Error ? exception.message : 'Failed to load campaign.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = searchParams?.get('workspace');
    if (initial) {
      setWorkspace(initial);
      loadCampaign(initial);
    }
  }, [loadCampaign, searchParams]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalized = workspace.trim();
    if (normalized) {
      const url = new URL(window.location.href);
      url.searchParams.set('workspace', normalized);
      window.history.replaceState({}, '', url);
    }
    loadCampaign(normalized);
  };

  return (
    <DashboardShell
      title="Optimization Campaign"
      subtitle="Inspect the incumbent, active directions, per-case gates, and evidence frontier."
      onRefresh={data ? () => loadCampaign(data.workspace) : undefined}
    >
      <section className="card">
        <form onSubmit={submit} className="card-body flex flex-col gap-3 md:flex-row md:items-end">
          <label className="flex-1 space-y-2">
            <span className="text-xs uppercase text-white/40">Campaign workspace</span>
            <input
              id="campaign-workspace"
              value={workspace}
              onChange={(event) => setWorkspace(event.target.value)}
              placeholder="artifacts/campaigns/launch-overhead"
              className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-sm text-white placeholder:text-white/30 focus:border-accent-primary/50 focus:outline-none"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/80 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Loading' : 'Load campaign'}
          </button>
        </form>
      </section>

      {error && (
        <section className="card">
          <div className="card-body text-sm text-accent-danger">{error}</div>
        </section>
      )}

      {data ? (
        <CampaignSnapshot data={data} />
      ) : !error && !loading ? (
        <section className="card">
          <div className="card-body text-sm text-white/50">
            Load a campaign workspace to review its append-only ledger.
          </div>
        </section>
      ) : null}
    </DashboardShell>
  );
}

export default function CampaignPage() {
  return (
    <Suspense fallback={<div className="p-8 text-white/60">Loading campaign view...</div>}>
      <CampaignPageInner />
    </Suspense>
  );
}
