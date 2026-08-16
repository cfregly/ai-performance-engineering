'use client';

import { useCallback, useDeferredValue, useEffect, useMemo, useState } from 'react';
import {
  Braces,
  Copy,
  Download,
  FileCode2,
  FileText,
  Layers3,
  Link2,
  RefreshCw,
  Waypoints,
  WandSparkles,
} from 'lucide-react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { getBenchmarkContracts, renderBenchmarkRun } from '@/lib/api';
import { cn } from '@/lib/utils';
import type {
  BenchmarkContractEntry,
  BenchmarkContractInterfaceEntry,
  BenchmarkContractsSummary,
  BenchmarkRunGeneratorDefaults,
  BenchmarkRunRenderResult,
} from '@/types';

type GeneratorForm = {
  name: string;
  benchmarkClass: 'publication_grade' | 'realism_grade';
  workloadType: 'training' | 'inference' | 'mixed';
  schedulerPath: string;
  cadence: 'canary' | 'nightly' | 'pre_release';
  model: string;
  precision: string;
  batchingPolicy: string;
  concurrencyModel: string;
  comparisonVariable:
    | 'hardware_generation'
    | 'runtime_version'
    | 'scheduler_path'
    | 'control_plane_path'
    | 'driver_stack'
    | 'network_topology'
    | 'storage_stack';
};

const FALLBACK_FORM: GeneratorForm = {
  name: 'publication-inference-stack-b200',
  benchmarkClass: 'publication_grade',
  workloadType: 'inference',
  schedulerPath: 'slinky-kueue',
  cadence: 'pre_release',
  model: 'openai/gpt-oss-20b',
  precision: 'bf16',
  batchingPolicy: 'continuous',
  concurrencyModel: 'closed_loop',
  comparisonVariable: 'runtime_version',
};

function fallbackInterfaceEntries(summary?: BenchmarkContractsSummary | null): BenchmarkContractInterfaceEntry[] {
  if (summary?.interface_entries?.length) {
    return summary.interface_entries;
  }
  return [
    {
      id: 'cli',
      label: 'CLI',
      transport: 'cli',
      entrypoint: summary?.interfaces.cli || 'python -m cli.aisp tools benchmark-contracts',
      description: 'Print the shared contract summary as JSON.',
    },
    {
      id: 'dashboard_api',
      label: 'Dashboard API',
      transport: 'http',
      entrypoint: summary?.interfaces.dashboard_api || '/api/benchmark/contracts',
      method: 'GET',
      description: 'Read-only route used by the contracts tab and other UI clients.',
    },
    {
      id: 'mcp',
      label: 'MCP',
      transport: 'mcp',
      entrypoint: summary?.interfaces.mcp_tool || 'benchmark_contracts',
      description: 'MCP tool exposing the same summary object to remote clients.',
    },
  ];
}

function normalizeGeneratorDefaults(summary?: BenchmarkContractsSummary | null): GeneratorForm {
  const defaults = summary?.generator?.defaults;
  if (!defaults) return FALLBACK_FORM;
  return {
    ...FALLBACK_FORM,
    ...(defaults as BenchmarkRunGeneratorDefaults),
  };
}

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'benchmark-run';
}

function ContractsSkeleton() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={`contracts-skel-${index}`} className="card p-5 animate-pulse">
          <div className="h-4 w-28 bg-white/10 rounded mb-4" />
          <div className="h-3 w-full bg-white/10 rounded mb-2" />
          <div className="h-3 w-5/6 bg-white/10 rounded mb-5" />
          <div className="space-y-2">
            <div className="h-7 bg-white/5 rounded-lg" />
            <div className="h-7 bg-white/5 rounded-lg" />
          </div>
        </div>
      ))}
    </div>
  );
}

function SurfaceCard({
  entry,
  onCopyPath,
  onCopyLink,
}: {
  entry: BenchmarkContractEntry;
  onCopyPath: (path: string) => void;
  onCopyLink: (anchor: string) => void;
}) {
  const summary = entry.summary;
  const anchor = `contract-${entry.name}`;

  return (
    <div id={anchor} className="card scroll-mt-24">
      <div className="card-header items-start gap-3">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-white/5 border border-white/10">
            {entry.kind === 'yaml' ? (
              <FileCode2 className="w-4 h-4 text-accent-primary" />
            ) : (
              <FileText className="w-4 h-4 text-accent-primary" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-base font-semibold text-white">{entry.name}</h2>
              <span className={cn('badge', entry.exists ? 'badge-success' : 'badge-danger')}>
                {entry.exists ? entry.kind : 'missing'}
              </span>
            </div>
            <p className="text-sm text-white/60 mt-1">{entry.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onCopyPath(entry.path)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
          >
            <span className="inline-flex items-center gap-2">
              <Copy className="w-3.5 h-3.5" />
              Path
            </span>
          </button>
          <button
            onClick={() => onCopyLink(anchor)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
          >
            <span className="inline-flex items-center gap-2">
              <Link2 className="w-3.5 h-3.5" />
              Link
            </span>
          </button>
        </div>
      </div>
      <div className="card-body space-y-4">
        <div className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Path</div>
          <div className="font-mono text-xs text-white/75 break-all">{entry.path}</div>
        </div>

        {summary && (
          <div className="space-y-3">
            <div>
              <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Top-level keys</div>
              <div className="flex flex-wrap gap-2">
                {summary.top_level_keys.map((key) => (
                  <span key={`${entry.name}-top-${key}`} className="badge badge-info">
                    {key}
                  </span>
                ))}
              </div>
            </div>

            {summary.spec_keys && summary.spec_keys.length > 0 && (
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Spec keys</div>
                <div className="flex flex-wrap gap-2">
                  {summary.spec_keys.map((key) => (
                    <span key={`${entry.name}-spec-${key}`} className="badge badge-warning">
                      {key}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {(summary.enabled_layers || summary.has_observability || summary.has_sinks) && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Layers</div>
                  <div className="text-sm text-white/80">
                    {summary.enabled_layers?.length ? summary.enabled_layers.join(', ') : '—'}
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Observability</div>
                  <div className={cn('text-sm font-medium', summary.has_observability ? 'text-accent-success' : 'text-white/50')}>
                    {summary.has_observability ? 'present' : '—'}
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Sinks</div>
                  <div className={cn('text-sm font-medium', summary.has_sinks ? 'text-accent-success' : 'text-white/50')}>
                    {summary.has_sinks ? 'present' : '—'}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ContractsPage() {
  const { showToast } = useToast();
  const [data, setData] = useState<BenchmarkContractsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<GeneratorForm | null>(null);
  const [yamlRender, setYamlRender] = useState<BenchmarkRunRenderResult | null>(null);
  const [yamlError, setYamlError] = useState<string | null>(null);
  const [renderingYaml, setRenderingYaml] = useState(false);

  const loadContracts = useCallback(async (isRefresh = false) => {
    try {
      if (!isRefresh) setLoading(true);
      setError(null);
      const payload = await getBenchmarkContracts();
      setData(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load benchmark contracts.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadContracts();
  }, [loadContracts]);

  useEffect(() => {
    if (!data) return;
    setForm((current) => current ?? normalizeGeneratorDefaults(data));
    setYamlRender((current) =>
      current
        ?? {
          schema_version: data.schema_version,
          generated_at_utc: data.generated_at_utc,
          template_path: data.generator.template_path,
          applied_values: normalizeGeneratorDefaults(data),
          rendered_yaml: data.generator.preview_yaml,
        }
    );
  }, [data]);

  useEffect(() => {
    if (!data || typeof window === 'undefined' || !window.location.hash) return;
    const target = document.getElementById(window.location.hash.slice(1));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [data]);

  const generatorDefaults = useMemo(() => normalizeGeneratorDefaults(data), [data]);
  const currentForm = form ?? generatorDefaults;
  const deferredForm = useDeferredValue(currentForm);
  const yamlPreview = yamlRender?.rendered_yaml || data?.generator.preview_yaml || '';

  useEffect(() => {
    if (!data) return;
    let cancelled = false;
    setRenderingYaml(true);
    setYamlError(null);
    renderBenchmarkRun(deferredForm)
      .then((result) => {
        if (!cancelled) {
          setYamlRender(result);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setYamlError(e instanceof Error ? e.message : 'Failed to render BenchmarkRun YAML.');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setRenderingYaml(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [data, deferredForm]);

  const copyText = useCallback(
    async (text: string, successMessage: string) => {
      try {
        await navigator.clipboard.writeText(text);
        showToast(successMessage, 'success');
      } catch (e) {
        showToast(e instanceof Error ? e.message : 'Failed to copy.', 'error');
      }
    },
    [showToast]
  );

  const copyDeepLink = useCallback(
    async (anchor: string) => {
      const url = `${window.location.origin}/contracts#${anchor}`;
      await copyText(url, 'Deep link copied to clipboard.');
    },
    [copyText]
  );

  const updateForm = useCallback(
    (patch: Partial<GeneratorForm>) => {
      setForm((current) => ({
        ...(current ?? generatorDefaults),
        ...patch,
      }));
    },
    [generatorDefaults]
  );

  const downloadYaml = useCallback(() => {
    const blob = new Blob([yamlPreview], { type: 'text/yaml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slugify(currentForm.name)}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('BenchmarkRun YAML downloaded.', 'success');
  }, [currentForm.name, showToast, yamlPreview]);

  const entries = useMemo(() => {
    if (!data) return [];
    const names = data.surface_order?.length ? data.surface_order : Object.keys(data.contracts || {});
    return names
      .map((name) => data.contracts[name])
      .filter((entry): entry is BenchmarkContractEntry => Boolean(entry));
  }, [data]);
  const interfaceEntries = useMemo(() => fallbackInterfaceEntries(data), [data]);
  const docCount = entries.filter((entry) => entry.kind === 'doc').length;
  const yamlCount = entries.filter((entry) => entry.kind === 'yaml').length;

  const actions = (
    <button
      onClick={() => {
        setRefreshing(true);
        loadContracts(true);
      }}
      className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
      disabled={refreshing}
    >
      <RefreshCw className={cn('w-4 h-4', refreshing ? 'animate-spin' : '')} />
      Refresh
    </button>
  );

  return (
    <DashboardShell
      title="Benchmark Contracts"
      subtitle="Methodology, warehouse, BenchmarkRun, and interface surfaces in one place."
      actions={actions}
    >
      {loading && !data ? (
        <ContractsSkeleton />
      ) : error && !data ? (
        <div className="card">
          <div className="card-body py-16 text-center">
            <p className="text-lg text-white/80">Failed to load contract surfaces.</p>
            <p className="text-sm text-white/50 mt-2">{error}</p>
          </div>
        </div>
      ) : (
        <>
          <section className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            <div id="interfaces" className="card xl:col-span-2 scroll-mt-24">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Waypoints className="w-5 h-5 text-accent-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Interface Surfaces</h2>
                    <p className="text-xs text-white/50">One shared contract, three thin entrypoints.</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-success">{data?.available ? 'available' : 'offline'}</span>
                  <button
                    onClick={() => copyDeepLink('interfaces')}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Link2 className="w-3.5 h-3.5" />
                      Link
                    </span>
                  </button>
                </div>
              </div>
              <div className="card-body grid grid-cols-1 gap-3">
                {interfaceEntries.map((entry) => (
                  <div key={entry.id} className="rounded-lg border border-white/10 bg-black/20 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 flex-wrap mb-2">
                          <div className="text-[11px] uppercase tracking-[0.2em] text-white/40">{entry.label}</div>
                          <span className="badge badge-info">{entry.transport}</span>
                          {entry.method && <span className="badge badge-warning">{entry.method}</span>}
                        </div>
                        <div className="font-mono text-sm text-white/85">{entry.entrypoint}</div>
                        <div className="text-xs text-white/50 mt-2">{entry.description}</div>
                      </div>
                      <button
                        onClick={() => copyText(entry.entrypoint, `${entry.label} entrypoint copied.`)}
                        className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Layers3 className="w-5 h-5 text-accent-secondary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Surface Count</h2>
                    <p className="text-xs text-white/50">Current exposed contract files.</p>
                  </div>
                </div>
              </div>
              <div className="card-body space-y-4">
                <div>
                  <div className="text-3xl font-semibold text-white">{data?.surface_count ?? entries.length}</div>
                  <div className="text-sm text-white/50">total surfaces</div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xl font-semibold text-white">{docCount}</div>
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mt-1">Docs</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xl font-semibold text-white">{yamlCount}</div>
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mt-1">YAML</div>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-xl font-semibold text-white">{data?.missing_surface_count ?? 0}</div>
                  <div className="text-xs text-white/50 uppercase tracking-[0.2em] mt-1">Missing</div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Braces className="w-5 h-5 text-accent-warning" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Contract Metadata</h2>
                    <p className="text-xs text-white/50">Schema, generation time, and repo source.</p>
                  </div>
                </div>
              </div>
              <div className="card-body space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mb-1">Schema</div>
                    <div className="font-mono text-xs text-white/80">{data?.schema_version}</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mb-1">Generated</div>
                    <div className="font-mono text-xs text-white/80 break-all">{data?.generated_at_utc}</div>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Repo root</div>
                  <div className="font-mono text-xs text-white/75 break-all">{data?.repo_root}</div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div id="generator" className="card scroll-mt-24">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <WandSparkles className="w-5 h-5 text-accent-success" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">BenchmarkRun Generator</h2>
                    <p className="text-xs text-white/50">Small form, valid YAML, copy/download ready.</p>
                  </div>
                </div>
                <button
                  onClick={() => copyDeepLink('generator')}
                  className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                >
                  <span className="inline-flex items-center gap-2">
                    <Link2 className="w-3.5 h-3.5" />
                    Link
                  </span>
                </button>
              </div>
              <div className="card-body space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Run Name</div>
                    <input
                      value={currentForm.name}
                      onChange={(e) => updateForm({ name: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Benchmark Class</div>
                    <select
                      value={currentForm.benchmarkClass}
                      onChange={(e) => updateForm({ benchmarkClass: e.target.value as GeneratorForm['benchmarkClass'] })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      {data?.generator?.choices.benchmarkClass?.map((choice) => (
                        <option key={choice} value={choice}>{choice}</option>
                      )) ?? (
                        <>
                          <option value="publication_grade">publication_grade</option>
                          <option value="realism_grade">realism_grade</option>
                        </>
                      )}
                    </select>
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Workload Type</div>
                    <select
                      value={currentForm.workloadType}
                      onChange={(e) => updateForm({ workloadType: e.target.value as GeneratorForm['workloadType'] })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      {data?.generator?.choices.workloadType?.map((choice) => (
                        <option key={choice} value={choice}>{choice}</option>
                      )) ?? (
                        <>
                          <option value="inference">inference</option>
                          <option value="training">training</option>
                          <option value="mixed">mixed</option>
                        </>
                      )}
                    </select>
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Cadence</div>
                    <select
                      value={currentForm.cadence}
                      onChange={(e) => updateForm({ cadence: e.target.value as GeneratorForm['cadence'] })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      {data?.generator?.choices.cadence?.map((choice) => (
                        <option key={choice} value={choice}>{choice}</option>
                      )) ?? (
                        <>
                          <option value="canary">canary</option>
                          <option value="nightly">nightly</option>
                          <option value="pre_release">pre_release</option>
                        </>
                      )}
                    </select>
                  </label>
                  <label className="space-y-1 md:col-span-2">
                    <div className="text-xs uppercase text-white/40">Scheduler Path</div>
                    <input
                      value={currentForm.schedulerPath}
                      onChange={(e) => updateForm({ schedulerPath: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1 md:col-span-2">
                    <div className="text-xs uppercase text-white/40">Model</div>
                    <input
                      value={currentForm.model}
                      onChange={(e) => updateForm({ model: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Precision</div>
                    <input
                      value={currentForm.precision}
                      onChange={(e) => updateForm({ precision: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Batching Policy</div>
                    <input
                      value={currentForm.batchingPolicy}
                      onChange={(e) => updateForm({ batchingPolicy: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Concurrency Model</div>
                    <input
                      value={currentForm.concurrencyModel}
                      onChange={(e) => updateForm({ concurrencyModel: e.target.value })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Variable Under Test</div>
                    <select
                      value={currentForm.comparisonVariable}
                      onChange={(e) => updateForm({ comparisonVariable: e.target.value as GeneratorForm['comparisonVariable'] })}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      {data?.generator?.choices.comparisonVariable?.map((choice) => (
                        <option key={choice} value={choice}>{choice}</option>
                      )) ?? (
                        <>
                          <option value="hardware_generation">hardware_generation</option>
                          <option value="runtime_version">runtime_version</option>
                          <option value="scheduler_path">scheduler_path</option>
                          <option value="control_plane_path">control_plane_path</option>
                          <option value="driver_stack">driver_stack</option>
                          <option value="network_topology">network_topology</option>
                          <option value="storage_stack">storage_stack</option>
                        </>
                      )}
                    </select>
                  </label>
                </div>

                <div className="rounded-lg border border-accent-info/20 bg-accent-info/10 px-4 py-3 text-sm text-white/75">
                  This starter follows the current repo contract metadata instead of a frontend-only default. It is still meant to be tightened against the actual workload and cluster before submission.
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Template source</div>
                  <div className="font-mono text-xs text-white/75 break-all">{data?.generator?.template_path}</div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <FileCode2 className="w-5 h-5 text-accent-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Generated YAML</h2>
                    <p className="text-xs text-white/50">Backend-rendered from the shared template source of truth.</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {renderingYaml && <span className="badge badge-info">rendering</span>}
                  <button
                    onClick={() => copyText(yamlPreview, 'BenchmarkRun YAML copied.')}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Copy className="w-3.5 h-3.5" />
                      Copy YAML
                    </span>
                  </button>
                  <button
                    onClick={downloadYaml}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Download className="w-3.5 h-3.5" />
                      Download
                    </span>
                  </button>
                </div>
              </div>
              <div className="card-body space-y-4">
                {yamlError && (
                  <div className="rounded-lg border border-accent-danger/20 bg-accent-danger/10 px-4 py-3 text-sm text-white/75">
                    {yamlError}
                  </div>
                )}
                <div className="rounded-lg border border-white/10 bg-black/40 p-4">
                  <pre className="font-mono text-xs text-white/80 overflow-x-auto whitespace-pre-wrap">{yamlPreview}</pre>
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Suggested next command</div>
                  <div className="font-mono text-xs text-white/80 break-all">
                    kubectl apply -f {slugify(currentForm.name)}.yaml
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {entries.map((entry) => (
              <SurfaceCard
                key={entry.name}
                entry={entry}
                onCopyPath={(path) => copyText(path, 'Contract path copied.')}
                onCopyLink={copyDeepLink}
              />
            ))}
          </section>
        </>
      )}
    </DashboardShell>
  );
}
