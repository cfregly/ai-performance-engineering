'use client';

import { useCallback, useMemo, useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { buildCanonicalPackage, promoteClusterRun, runClusterCommonEval, runClusterEvalSuite, validateFieldReport } from '@/lib/api';
import { cn } from '@/lib/utils';
import type {
  ClusterCanonicalPackageResult,
  ClusterCommonEvalResult,
  ClusterEvalSuiteResult,
  FieldReportValidationResult,
  ClusterPromoteRunResult,
} from '@/types';
import { Archive, FileCheck2, RefreshCw, Server } from 'lucide-react';

function splitList(value: string): string[] | undefined {
  const items = value
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0);
  return items.length ? items : undefined;
}

function badge(ok: boolean | undefined) {
  if (ok) return 'badge badge-success';
  return 'badge badge-danger';
}

function codeBlock(text: string | undefined | null) {
  if (!text) return null;
  return (
    <pre className="text-xs text-white/80 bg-black/30 border border-white/10 rounded-lg p-4 overflow-x-auto">
      {text}
    </pre>
  );
}

export default function ClusterPage() {
  const { showToast } = useToast();

  const [commonPreset, setCommonPreset] = useState<'common-answer-fast' | 'core-system' | 'modern-llm' | 'multinode-readiness'>('common-answer-fast');
  const [coverageBaselineRunId, setCoverageBaselineRunId] = useState<string>('');
  const [runningCommonEval, setRunningCommonEval] = useState(false);
  const [commonEvalResult, setCommonEvalResult] = useState<ClusterCommonEvalResult | null>(null);

  const [mode, setMode] = useState<'smoke' | 'full'>('smoke');
  const [runId, setRunId] = useState<string>('');
  const [hosts, setHosts] = useState<string>('');
  const [labels, setLabels] = useState<string>('');
  const [sshUser, setSshUser] = useState<string>('');
  const [sshKey, setSshKey] = useState<string>('');
  const [oobIf, setOobIf] = useState<string>('');
  const [socketIfname, setSocketIfname] = useState<string>('');
  const [ncclIbHca, setNcclIbHca] = useState<string>('');
  const [primaryLabel, setPrimaryLabel] = useState<string>('');
  const [extraArgs, setExtraArgs] = useState<string>('');
  const [timeoutSeconds, setTimeoutSeconds] = useState<string>('');

  const [runningSuite, setRunningSuite] = useState(false);
  const [suiteResult, setSuiteResult] = useState<ClusterEvalSuiteResult | null>(null);

  const [reportPath, setReportPath] = useState('cluster/field-report.md');
  const [notesPath, setNotesPath] = useState('cluster/field-report-notes.md');
  const [templatePath, setTemplatePath] = useState('cluster/docs/field-report-template.md');
  const [runbookPath, setRunbookPath] = useState('cluster/docs/advanced-runbook.md');
  const [canonicalRunId, setCanonicalRunId] = useState<string>('');
  const [allowRunId, setAllowRunId] = useState<string>('');
  const [runningValidator, setRunningValidator] = useState(false);
  const [validatorResult, setValidatorResult] = useState<FieldReportValidationResult | null>(null);

  const [packageCanonicalRunId, setPackageCanonicalRunId] = useState<string>('');
  const [packageComparisonRunIds, setPackageComparisonRunIds] = useState<string>('');
  const [packageHistoricalRunIds, setPackageHistoricalRunIds] = useState<string>('');
  const [packageOutputDir, setPackageOutputDir] = useState<string>('cluster/canonical_package');
  const [packageTimeoutSeconds, setPackageTimeoutSeconds] = useState<string>('300');
  const [runningPackage, setRunningPackage] = useState(false);
  const [packageResult, setPackageResult] = useState<ClusterCanonicalPackageResult | null>(null);

  const [promoteRunId, setPromoteRunId] = useState<string>('');
  const [promoteLabel, setPromoteLabel] = useState<string>('localhost');
  const [promoteAllowRunIds, setPromoteAllowRunIds] = useState<string>('');
  const [promoteCleanup, setPromoteCleanup] = useState<boolean>(false);
  const [promoteSkipRender, setPromoteSkipRender] = useState<boolean>(false);
  const [promoteSkipValidate, setPromoteSkipValidate] = useState<boolean>(false);
  const [runningPromote, setRunningPromote] = useState(false);
  const [promoteResult, setPromoteResult] = useState<ClusterPromoteRunResult | null>(null);

  const suiteMeta = useMemo(() => {
    if (!suiteResult) return null;
    const modeValue = suiteResult.mode || '—';
    const idValue = suiteResult.run_id || '—';
    return `${modeValue} • ${idValue}`;
  }, [suiteResult]);

  const commonEvalMeta = useMemo(() => {
    if (!commonEvalResult) return null;
    const presetValue = commonEvalResult.preset || '—';
    const idValue = commonEvalResult.run_id || '—';
    return `${presetValue} • ${idValue}`;
  }, [commonEvalResult]);

  const runCommonEval = useCallback(async () => {
    try {
      setRunningCommonEval(true);
      setCommonEvalResult(null);

      const params: Record<string, unknown> = {
        preset: commonPreset,
      };
      if (runId.trim()) params.run_id = runId.trim();
      if (timeoutSeconds.trim()) params.timeout_seconds = Number(timeoutSeconds);

      const hostsList = splitList(hosts);
      const labelsList = splitList(labels);
      const extraArgsList = splitList(extraArgs);

      if (hostsList) params.hosts = hostsList;
      if (labelsList) params.labels = labelsList;
      if (sshUser.trim()) params.ssh_user = sshUser.trim();
      if (sshKey.trim()) params.ssh_key = sshKey.trim();
      if (oobIf.trim()) params.oob_if = oobIf.trim();
      if (socketIfname.trim()) params.socket_ifname = socketIfname.trim();
      if (ncclIbHca.trim()) params.nccl_ib_hca = ncclIbHca.trim();
      if (primaryLabel.trim()) params.primary_label = primaryLabel.trim();
      if (coverageBaselineRunId.trim()) params.coverage_baseline_run_id = coverageBaselineRunId.trim();
      if (extraArgsList) params.extra_args = extraArgsList;

      const data = (await runClusterCommonEval(params)) as ClusterCommonEvalResult;
      setCommonEvalResult(data);
      if (data.success) {
        showToast('Common eval completed successfully.', 'success');
      } else {
        showToast(data.error || 'Common eval failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Common eval failed.', 'error');
      setCommonEvalResult(null);
    } finally {
      setRunningCommonEval(false);
    }
  }, [
    commonPreset,
    coverageBaselineRunId,
    extraArgs,
    hosts,
    labels,
    ncclIbHca,
    oobIf,
    primaryLabel,
    runId,
    showToast,
    socketIfname,
    sshKey,
    sshUser,
    timeoutSeconds,
  ]);

  const runSuite = useCallback(async () => {
    try {
      setRunningSuite(true);
      setSuiteResult(null);
      const params: Record<string, unknown> = {
        mode,
      };
      if (runId.trim()) params.run_id = runId.trim();
      if (timeoutSeconds.trim()) params.timeout_seconds = Number(timeoutSeconds);

      const hostsList = splitList(hosts);
      const labelsList = splitList(labels);
      const extraArgsList = splitList(extraArgs);

      if (hostsList) params.hosts = hostsList;
      if (labelsList) params.labels = labelsList;
      if (sshUser.trim()) params.ssh_user = sshUser.trim();
      if (sshKey.trim()) params.ssh_key = sshKey.trim();
      if (oobIf.trim()) params.oob_if = oobIf.trim();
      if (socketIfname.trim()) params.socket_ifname = socketIfname.trim();
      if (ncclIbHca.trim()) params.nccl_ib_hca = ncclIbHca.trim();
      if (primaryLabel.trim()) params.primary_label = primaryLabel.trim();
      if (extraArgsList) params.extra_args = extraArgsList;

      const data = (await runClusterEvalSuite(params)) as ClusterEvalSuiteResult;
      setSuiteResult(data);
      if (data.success) {
        showToast('Cluster eval suite completed successfully.', 'success');
      } else {
        showToast(data.error || 'Cluster eval suite failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Cluster eval suite failed.', 'error');
      setSuiteResult(null);
    } finally {
      setRunningSuite(false);
    }
  }, [
    extraArgs,
    hosts,
    labels,
    mode,
    ncclIbHca,
    oobIf,
    primaryLabel,
    runId,
    showToast,
    socketIfname,
    sshKey,
    sshUser,
    timeoutSeconds,
  ]);

  const runValidator = useCallback(async () => {
    try {
      setRunningValidator(true);
      setValidatorResult(null);
      const params: Record<string, unknown> = {};
      if (reportPath.trim()) params.report = reportPath.trim();
      if (notesPath.trim()) params.notes = notesPath.trim();
      if (templatePath.trim()) params.template = templatePath.trim();
      if (runbookPath.trim()) params.runbook = runbookPath.trim();
      if (canonicalRunId.trim()) params.canonical_run_id = canonicalRunId.trim();
      const allow = splitList(allowRunId);
      if (allow) params.allow_run_id = allow;

      const data = (await validateFieldReport(params)) as FieldReportValidationResult;
      setValidatorResult(data);
      if (data.success) {
        showToast('Field report validation passed.', 'success');
      } else {
        showToast(data.error || 'Field report validation failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Field report validation failed.', 'error');
      setValidatorResult(null);
    } finally {
      setRunningValidator(false);
    }
  }, [
    allowRunId,
    canonicalRunId,
    notesPath,
    reportPath,
    runbookPath,
    showToast,
    templatePath,
  ]);

  const runPackage = useCallback(async () => {
    try {
      setRunningPackage(true);
      setPackageResult(null);

      const canonicalRunId = packageCanonicalRunId.trim();
      if (!canonicalRunId) {
        showToast('canonical_run_id is required.', 'warning');
        return;
      }

      const params: Record<string, unknown> = {
        canonical_run_id: canonicalRunId,
      };
      const comparison = splitList(packageComparisonRunIds);
      const historical = splitList(packageHistoricalRunIds);
      if (comparison) params.comparison_run_ids = comparison;
      if (historical) params.historical_run_ids = historical;
      if (packageOutputDir.trim()) params.output_dir = packageOutputDir.trim();
      if (packageTimeoutSeconds.trim()) params.timeout_seconds = Number(packageTimeoutSeconds);

      const data = (await buildCanonicalPackage(params)) as ClusterCanonicalPackageResult;
      setPackageResult(data);
      if (data.success) {
        showToast('Canonical package built successfully.', 'success');
      } else {
        showToast(data.error || 'Canonical package build failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Canonical package build failed.', 'error');
      setPackageResult(null);
    } finally {
      setRunningPackage(false);
    }
  }, [
    packageCanonicalRunId,
    packageComparisonRunIds,
    packageHistoricalRunIds,
    packageOutputDir,
    packageTimeoutSeconds,
    showToast,
  ]);

  const runPromote = useCallback(async () => {
    try {
      setRunningPromote(true);
      setPromoteResult(null);

      const promotedRunId = promoteRunId.trim();
      if (!promotedRunId) {
        showToast('run_id is required for promotion.', 'warning');
        return;
      }

      const params: Record<string, unknown> = {
        run_id: promotedRunId,
        label: promoteLabel.trim() || 'localhost',
        cleanup: promoteCleanup,
        skip_render_localhost_report: promoteSkipRender,
        skip_validate_localhost_report: promoteSkipValidate,
      };
      const allow = splitList(promoteAllowRunIds);
      if (allow) params.allow_run_ids = allow;

      const data = (await promoteClusterRun(params)) as ClusterPromoteRunResult;
      setPromoteResult(data);
      if (data.success) {
        showToast('Run promoted successfully.', 'success');
      } else {
        showToast(data.error || 'Run promotion failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Run promotion failed.', 'error');
      setPromoteResult(null);
    } finally {
      setRunningPromote(false);
    }
  }, [
    promoteAllowRunIds,
    promoteCleanup,
    promoteLabel,
    promoteRunId,
    promoteSkipRender,
    promoteSkipValidate,
    showToast,
  ]);

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Cluster evaluation workflows under cluster/ (field report, discovery, reproducibility)."
      actions={
        <button
          onClick={runCommonEval}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
          disabled={runningCommonEval}
        >
          <RefreshCw className={cn('w-4 h-4', runningCommonEval ? 'animate-spin' : '')} />
          Run Common Eval
        </button>
      }
    >
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Server className="w-5 h-5 text-accent-info" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Common Eval</h2>
                <p className="text-xs text-white/50">
                  Recommended preset entrypoint for the standard benchmark bundles people ask for first.
                </p>
              </div>
            </div>
            {commonEvalResult && (
              <span className={badge(commonEvalResult.success)}>
                {commonEvalResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">Preset</div>
                <select
                  value={commonPreset}
                  onChange={(e) => setCommonPreset(e.target.value as 'common-answer-fast' | 'core-system' | 'modern-llm' | 'multinode-readiness')}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                >
                  <option value="common-answer-fast">common-answer-fast</option>
                  <option value="core-system">core-system</option>
                  <option value="modern-llm">modern-llm</option>
                  <option value="multinode-readiness">multinode-readiness</option>
                </select>
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">RUN_ID</div>
                <input
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                  placeholder="(optional) default: YYYY-MM-DD"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">Hosts</div>
                <input
                  value={hosts}
                  onChange={(e) => setHosts(e.target.value)}
                  placeholder="comma-separated host list"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">Labels (optional)</div>
                <input
                  value={labels}
                  onChange={(e) => setLabels(e.target.value)}
                  placeholder="comma-separated, same count as hosts"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">SSH User</div>
                <input
                  value={sshUser}
                  onChange={(e) => setSshUser(e.target.value)}
                  placeholder="(optional)"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">SSH Key</div>
                <input
                  value={sshKey}
                  onChange={(e) => setSshKey(e.target.value)}
                  placeholder="(optional) /path/to/key"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">oob_if</div>
                <input
                  value={oobIf}
                  onChange={(e) => setOobIf(e.target.value)}
                  placeholder="(optional) required for multi-node readiness"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">socket_ifname</div>
                <input
                  value={socketIfname}
                  onChange={(e) => setSocketIfname(e.target.value)}
                  placeholder="(optional) e.g. eth0"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">NCCL_IB_HCA</div>
                <input
                  value={ncclIbHca}
                  onChange={(e) => setNcclIbHca(e.target.value)}
                  placeholder="(optional) e.g. mlx5_0"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">primary_label</div>
                <input
                  value={primaryLabel}
                  onChange={(e) => setPrimaryLabel(e.target.value)}
                  placeholder="(optional) for smoke/local steps"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">timeout_seconds</div>
                <input
                  value={timeoutSeconds}
                  onChange={(e) => setTimeoutSeconds(e.target.value)}
                  placeholder="(optional)"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">coverage_baseline_run_id (optional)</div>
                <input
                  value={coverageBaselineRunId}
                  onChange={(e) => setCoverageBaselineRunId(e.target.value)}
                  placeholder="RUN_ID for coverage delta output"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">extra_args (optional)</div>
                <input
                  value={extraArgs}
                  onChange={(e) => setExtraArgs(e.target.value)}
                  placeholder="comma-separated args appended after preset defaults"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
            </div>

            <button
              onClick={runCommonEval}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningCommonEval}
            >
              <RefreshCw className={cn('w-4 h-4', runningCommonEval ? 'animate-spin' : '')} />
              {runningCommonEval ? 'Running...' : 'Run Common Eval'}
            </button>

            {commonEvalResult && (
              <div className="space-y-3">
                <div className="text-xs text-white/50">{commonEvalMeta}</div>
                {commonEvalResult.preset_description && (
                  <div className="text-xs text-white/70">{commonEvalResult.preset_description}</div>
                )}
                {commonEvalResult.run_dir && (
                  <div className="text-xs text-white/70">
                    run_dir: <span className="font-mono text-white/90">{commonEvalResult.run_dir}</span>
                  </div>
                )}
                {commonEvalResult.manifest_path && (
                  <div className="text-xs text-white/70">
                    manifest: <span className="font-mono text-white/90">{commonEvalResult.manifest_path}</span>
                  </div>
                )}
                {commonEvalResult.coverage_baseline_run_id && (
                  <div className="text-xs text-white/70">
                    coverage baseline: <span className="font-mono text-white/90">{commonEvalResult.coverage_baseline_run_id}</span>
                  </div>
                )}
                {commonEvalResult.artifact_roles && commonEvalResult.artifact_roles.length > 0 && (
                  <div className="text-xs text-white/70">
                    artifacts: <span className="font-mono text-white/90">{commonEvalResult.artifact_roles.join(', ')}</span>
                  </div>
                )}
                {commonEvalResult.error && (
                  <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                    {commonEvalResult.error}
                  </div>
                )}
                {codeBlock(commonEvalResult.stdout)}
                {codeBlock(commonEvalResult.stderr)}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <FileCheck2 className="w-5 h-5 text-accent-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Raw Eval Suite</h2>
                <p className="text-xs text-white/50">
                  Direct suite access for smoke/full mode and manual flag composition.
                </p>
              </div>
            </div>
            {suiteResult && (
              <span className={badge(suiteResult.success)}>
                {suiteResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">Mode</div>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value as 'smoke' | 'full')}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                >
                  <option value="smoke">smoke</option>
                  <option value="full">full</option>
                </select>
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">run_id (optional)</div>
                <input
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                  placeholder="default: YYYY-MM-DD"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">extra_args (optional)</div>
                <input
                  value={extraArgs}
                  onChange={(e) => setExtraArgs(e.target.value)}
                  placeholder="comma-separated args appended to run_cluster_eval_suite.sh"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
            </div>

            <button
              onClick={runSuite}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningSuite}
            >
              <RefreshCw className={cn('w-4 h-4', runningSuite ? 'animate-spin' : '')} />
              {runningSuite ? 'Running...' : 'Run Raw Eval Suite'}
            </button>

            {suiteResult?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {suiteResult.error}
              </div>
            )}
            {suiteResult && (
              <div className="space-y-3">
                <div className="text-xs text-white/50">{suiteMeta}</div>
                {suiteResult.run_dir && (
                  <div className="text-xs text-white/70">
                    run_dir: <span className="font-mono text-white/90">{suiteResult.run_dir}</span>
                  </div>
                )}
                {codeBlock(suiteResult.stdout || suiteResult.collect?.stdout)}
                {codeBlock(suiteResult.stderr || suiteResult.collect?.stderr)}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Archive className="w-5 h-5 text-accent-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Promote Run</h2>
                <p className="text-xs text-white/50">
                  Explicit publication step: copy one run into the published flat surfaces and localhost report package.
                </p>
              </div>
            </div>
            {promoteResult && (
              <span className={badge(promoteResult.success)}>
                {promoteResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">run_id</div>
                <input
                  value={promoteRunId}
                  onChange={(e) => setPromoteRunId(e.target.value)}
                  placeholder="required run id under cluster/runs/"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">label</div>
                <input
                  value={promoteLabel}
                  onChange={(e) => setPromoteLabel(e.target.value)}
                  placeholder="localhost"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">allow_run_ids</div>
                <input
                  value={promoteAllowRunIds}
                  onChange={(e) => setPromoteAllowRunIds(e.target.value)}
                  placeholder="comma-separated baselines to retain during cleanup/validation"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="flex items-center gap-3 text-sm text-white/80">
                <input type="checkbox" checked={promoteCleanup} onChange={(e) => setPromoteCleanup(e.target.checked)} />
                cleanup superseded artifacts + run dirs
              </label>
              <label className="flex items-center gap-3 text-sm text-white/80">
                <input type="checkbox" checked={promoteSkipRender} onChange={(e) => setPromoteSkipRender(e.target.checked)} />
                skip localhost report render
              </label>
              <label className="flex items-center gap-3 text-sm text-white/80">
                <input type="checkbox" checked={promoteSkipValidate} onChange={(e) => setPromoteSkipValidate(e.target.checked)} />
                skip localhost report validation
              </label>
            </div>

            <button
              onClick={runPromote}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningPromote}
            >
              <RefreshCw className={cn('w-4 h-4', runningPromote ? 'animate-spin' : '')} />
              {runningPromote ? 'Promoting...' : 'Promote Run'}
            </button>

            {promoteResult?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {promoteResult.error}
              </div>
            )}
            {promoteResult && (
              <div className="space-y-3">
                {promoteResult.run_dir && (
                  <div className="text-xs text-white/70">
                    run_dir: <span className="font-mono text-white/90">{promoteResult.run_dir}</span>
                  </div>
                )}
                {promoteResult.published_localhost_report_path && (
                  <div className="text-xs text-white/70">
                    report: <span className="font-mono text-white/90">{promoteResult.published_localhost_report_path}</span>
                  </div>
                )}
                {promoteResult.published_root && (
                  <div className="text-xs text-white/70">
                    published_root: <span className="font-mono text-white/90">{promoteResult.published_root}</span>
                  </div>
                )}
                {codeBlock(promoteResult.stdout)}
                {codeBlock(promoteResult.stderr)}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Archive className="w-5 h-5 text-accent-info" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Canonical Package</h2>
                <p className="text-xs text-white/50">
                  Builds one agent-friendly package from the selected live runs without touching source artifacts.
                </p>
              </div>
            </div>
            {packageResult && (
              <span className={badge(packageResult.success)}>
                {packageResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">canonical_run_id</div>
                <input
                  value={packageCanonicalRunId}
                  onChange={(e) => setPackageCanonicalRunId(e.target.value)}
                  placeholder="required primary run id"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">comparison_run_ids</div>
                <input
                  value={packageComparisonRunIds}
                  onChange={(e) => setPackageComparisonRunIds(e.target.value)}
                  placeholder="comma-separated optional baseline run ids"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">historical_run_ids</div>
                <input
                  value={packageHistoricalRunIds}
                  onChange={(e) => setPackageHistoricalRunIds(e.target.value)}
                  placeholder="comma-separated historical run ids to preserve"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">output_dir</div>
                <input
                  value={packageOutputDir}
                  onChange={(e) => setPackageOutputDir(e.target.value)}
                  placeholder="cluster/canonical_package"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">timeout_seconds</div>
                <input
                  value={packageTimeoutSeconds}
                  onChange={(e) => setPackageTimeoutSeconds(e.target.value)}
                  placeholder="300"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
            </div>

            <button
              onClick={runPackage}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningPackage}
            >
              <RefreshCw className={cn('w-4 h-4', runningPackage ? 'animate-spin' : '')} />
              {runningPackage ? 'Building...' : 'Build Canonical Package'}
            </button>

            {packageResult?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {packageResult.error}
              </div>
            )}
            {packageResult && (
              <div className="space-y-3">
                {packageResult.output_dir && (
                  <div className="text-xs text-white/70">
                    output: <span className="font-mono text-white/90">{packageResult.output_dir}</span>
                  </div>
                )}
                {packageResult.package_manifest_path && (
                  <div className="text-xs text-white/70">
                    manifest: <span className="font-mono text-white/90">{packageResult.package_manifest_path}</span>
                  </div>
                )}
                {packageResult.package_readme_path && (
                  <div className="text-xs text-white/70">
                    readme: <span className="font-mono text-white/90">{packageResult.package_readme_path}</span>
                  </div>
                )}
                {packageResult.cleanup_keep_run_ids_path && (
                  <div className="text-xs text-white/70">
                    keep-list: <span className="font-mono text-white/90">{packageResult.cleanup_keep_run_ids_path}</span>
                  </div>
                )}
                {codeBlock(packageResult.stdout)}
                {codeBlock(packageResult.stderr)}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <FileCheck2 className="w-5 h-5 text-accent-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Field Report Validator</h2>
                <p className="text-xs text-white/50">
                  Validates required sections plus artifact hygiene.
                </p>
              </div>
            </div>
            {validatorResult && (
              <span className={badge(validatorResult.success)}>
                {validatorResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">report</div>
                <input
                  value={reportPath}
                  onChange={(e) => setReportPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">notes</div>
                <input
                  value={notesPath}
                  onChange={(e) => setNotesPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">template</div>
                <input
                  value={templatePath}
                  onChange={(e) => setTemplatePath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">runbook</div>
                <input
                  value={runbookPath}
                  onChange={(e) => setRunbookPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">canonical_run_id (optional)</div>
                <input
                  value={canonicalRunId}
                  onChange={(e) => setCanonicalRunId(e.target.value)}
                  placeholder="RUN_ID expected by the report"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">allow_run_id (optional)</div>
                <input
                  value={allowRunId}
                  onChange={(e) => setAllowRunId(e.target.value)}
                  placeholder="comma-separated extra run ids allowed for hygiene checks"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
                />
              </label>
            </div>

            <button
              onClick={runValidator}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningValidator}
            >
              <RefreshCw className={cn('w-4 h-4', runningValidator ? 'animate-spin' : '')} />
              {runningValidator ? 'Validating...' : 'Validate Field Report'}
            </button>

            {validatorResult?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {validatorResult.error}
              </div>
            )}
            {validatorResult && (
              <div className="space-y-3">
                {codeBlock(validatorResult.stdout)}
                {codeBlock(validatorResult.stderr)}
              </div>
            )}
          </div>
        </div>
      </section>
    </DashboardShell>
  );
}
