'use client';

import type {
  CampaignArtifact,
  CampaignDashboardResult,
  CampaignExperimentSummary,
} from '@/types';
import { AlertTriangle, ExternalLink, GitBranch, Trophy } from 'lucide-react';

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'Not measured';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

function formatConfidence(value: number[] | null | undefined): string {
  if (!value || value.length < 2) return 'Not required';
  return `${formatPct(value[0])} to ${formatPct(value[1])}`;
}

function decisionClass(decision: string): string {
  if (decision === 'promote') return 'bg-accent-success/20 text-accent-success';
  if (decision === 'reject') return 'bg-accent-danger/20 text-accent-danger';
  if (decision === 'park') return 'bg-accent-warning/20 text-accent-warning';
  return 'bg-white/10 text-white/60';
}

function ArtifactLink({
  artifact,
  workspace,
}: {
  artifact: CampaignArtifact;
  workspace: string;
}) {
  const label = artifact.path.split('/').pop() || artifact.path;
  if (!artifact.downloadable) {
    return <span className="text-white/40" title={artifact.path}>{label}</span>;
  }
  const href = `/api/optimization/campaign/artifact?workspace=${encodeURIComponent(
    workspace
  )}&artifact=${encodeURIComponent(artifact.path)}`;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 text-accent-primary hover:underline"
      title={artifact.path}
    >
      {label}
      <ExternalLink className="h-3 w-3" />
    </a>
  );
}

function ExperimentBadge({ experiment }: { experiment: CampaignExperimentSummary }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-mono text-sm text-white">{experiment.experiment_id}</span>
      {experiment.is_incumbent && (
        <span className="rounded-full bg-accent-success/20 px-2 py-0.5 text-xs text-accent-success">
          Incumbent
        </span>
      )}
      <span className={`rounded-full px-2 py-0.5 text-xs ${decisionClass(experiment.gate.decision)}`}>
        {experiment.gate.decision}
      </span>
      <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-white/60">
        {experiment.beam}
      </span>
    </div>
  );
}

export function CampaignSnapshot({ data }: { data: CampaignDashboardResult }) {
  const latest = data.latest_measured;
  const incumbentCommitText = data.incumbent.commit.slice(0, 12);

  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="card">
          <div className="card-body">
            <div className="text-xs uppercase text-white/40">Incumbent</div>
            <div className="mt-2 text-xl font-semibold text-white">
              {data.incumbent.experiment_id || 'Frozen control'}
            </div>
            <div className="mt-1 font-mono text-xs text-white/40">
              {incumbentCommitText}
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <div className="text-xs uppercase text-white/40">Experiment ledger</div>
            <div className="mt-2 text-xl font-semibold text-white">{data.counts.experiments}</div>
            <div className="mt-1 text-xs text-white/40">{data.counts.measured} measured</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <div className="text-xs uppercase text-white/40">Best frontier</div>
            <div className="mt-2 text-xl font-semibold text-white">{data.frontier.length}</div>
            <div className="mt-1 text-xs text-white/40">Gate-passing steps</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <div className="text-xs uppercase text-white/40">Campaign budget</div>
            <div className={`mt-2 text-xl font-semibold ${data.budget.can_schedule ? 'text-accent-success' : 'text-accent-danger'}`}>
              {data.budget.can_schedule ? 'Open' : 'Exhausted'}
            </div>
            <div className="mt-1 text-xs text-white/40">
              {data.budget.completed_experiments} completed
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="text-lg font-semibold text-white">Active idea beam</h2>
              <p className="text-xs text-white/50">One live candidate per direction.</p>
            </div>
            <GitBranch className="h-5 w-5 text-white/40" />
          </div>
          <div className="card-body space-y-4">
            {data.active_beam.length === 0 ? (
              <p className="text-sm text-white/50">No active candidates.</p>
            ) : (
              data.active_beam.map((experiment) => (
                <div key={experiment.experiment_id} className="rounded-lg border border-white/5 bg-white/[0.02] p-4">
                  <ExperimentBadge experiment={experiment} />
                  <p className="mt-2 text-sm text-white/70">{experiment.hypothesis}</p>
                  <p className="mt-2 text-xs text-white/40">
                    Next: {experiment.next_step || 'Record the next falsification step'}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="text-lg font-semibold text-white">Best-so-far frontier</h2>
              <p className="text-xs text-white/50">Candidates that moved the passing frontier.</p>
            </div>
            <Trophy className="h-5 w-5 text-accent-warning" />
          </div>
          <div className="card-body space-y-4">
            {data.frontier.length === 0 ? (
              <p className="text-sm text-white/50">No candidate has passed every gate.</p>
            ) : (
              data.frontier.map((experiment) => (
                <div key={experiment.experiment_id} className="flex items-start justify-between gap-4 rounded-lg border border-white/5 bg-white/[0.02] p-4">
                  <div>
                    <ExperimentBadge experiment={experiment} />
                    <p className="mt-2 text-xs text-white/50">{experiment.hypothesis}</p>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold text-accent-success">
                      {formatPct(experiment.gate.improvement_pct)}
                    </div>
                    <div className="mt-1 text-xs text-white/40">
                      {formatConfidence(experiment.gate.improvement_ci_pct)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-lg font-semibold text-white">Latest promotion glance</h2>
            <p className="text-xs text-white/50">Per-case medians, confidence bounds, and frozen-case checks.</p>
          </div>
          {latest && <ExperimentBadge experiment={latest} />}
        </div>
        {!latest ? (
          <div className="card-body text-sm text-white/50">No measured experiment has been recorded.</div>
        ) : (
          <>
            {latest.gate.reasons.length > 0 && (
              <div className="mx-5 mt-5 rounded-lg border border-accent-warning/30 bg-accent-warning/10 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-accent-warning">
                  <AlertTriangle className="h-4 w-4" />
                  Gate findings
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-white/60">
                  {latest.gate.reasons.map((reason) => <li key={reason}>{reason}</li>)}
                </ul>
              </div>
            )}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5 text-xs uppercase text-white/50">
                    <th className="px-5 py-3 text-left">Case</th>
                    <th className="px-5 py-3 text-right">Control</th>
                    <th className="px-5 py-3 text-right">Candidate</th>
                    <th className="px-5 py-3 text-right">Improvement</th>
                    <th className="px-5 py-3 text-left">Confidence interval</th>
                    <th className="px-5 py-3 text-right">Trials</th>
                  </tr>
                </thead>
                <tbody>
                  {latest.cases.map((row) => (
                    <tr key={row.case_id} className="border-b border-white/5 text-sm">
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-2 text-white">
                          <span>{row.case_id}</span>
                          {row.primary && <span className="rounded bg-accent-primary/20 px-1.5 py-0.5 text-[10px] text-accent-primary">Primary</span>}
                          {row.frozen && <span className={`rounded px-1.5 py-0.5 text-[10px] ${row.frozen_case_violation ? 'bg-accent-danger/20 text-accent-danger' : 'bg-white/10 text-white/50'}`}>Frozen</span>}
                        </div>
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-white/70">{row.control_median.toPrecision(5)}</td>
                      <td className="px-5 py-3 text-right font-mono text-white/70">{row.candidate_median.toPrecision(5)}</td>
                      <td className={`px-5 py-3 text-right font-semibold ${row.frozen_case_violation || row.improvement_pct < 0 ? 'text-accent-danger' : 'text-accent-success'}`}>
                        {formatPct(row.improvement_pct)}
                      </td>
                      <td className="px-5 py-3 text-xs text-white/50">{formatConfidence(row.improvement_ci_pct)}</td>
                      <td className="px-5 py-3 text-right text-white/50">{Math.min(row.control_trials, row.candidate_trials)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card-body border-t border-white/5">
              <div className="text-xs uppercase text-white/40">Evidence artifacts</div>
              <div className="mt-2 flex flex-wrap gap-3 text-xs">
                {latest.artifacts.length === 0 ? (
                  <span className="text-white/40">No artifact references.</span>
                ) : (
                  latest.artifacts.map((artifact) => (
                    <ArtifactLink
                      key={`${artifact.role}:${artifact.path}`}
                      artifact={artifact}
                      workspace={data.workspace}
                    />
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
