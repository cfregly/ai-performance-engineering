import { render, screen } from '@testing-library/react';
import { CampaignSnapshot } from '@/components/CampaignSnapshot';
import type { CampaignDashboardResult, CampaignExperimentSummary } from '@/types';

const experiment: CampaignExperimentSummary = {
  experiment_id: 'exp-002',
  parent_id: 'exp-001',
  beam: 'launch-structure',
  hypothesis: 'Reduce replay launch overhead.',
  status: 'promoted',
  correctness: 'passed',
  primary_case: 'batch-1',
  changed_surface: ['labs/demo/optimized_demo.py'],
  mechanism: 'Fewer launches',
  outcome: 'Lower median latency',
  next_step: 'Profile the next shape',
  recorded_at: '2026-08-16T00:00:00+00:00',
  revision: 2,
  is_incumbent: true,
  gate: {
    decision: 'promote',
    reasons: [],
    improvement_pct: 8,
    improvement_ci_pct: [5, 11],
    confidence_required: true,
  },
  cases: [
    {
      case_id: 'batch-1',
      primary: true,
      frozen: true,
      control_median: 10,
      candidate_median: 9.2,
      improvement_pct: 8,
      improvement_ci_pct: [5, 11],
      control_trials: 5,
      candidate_trials: 5,
      frozen_case_violation: false,
    },
  ],
  artifacts: [
    {
      role: 'profile',
      path: 'artifacts/exp-002/candidate.ncu-rep',
      sha256: 'a'.repeat(64),
      downloadable: true,
    },
  ],
  provenance: { candidate_commit: '1234567890abcdef' },
};

const data: CampaignDashboardResult = {
  workspace: '/tmp/campaign',
  config: {},
  budget: {
    completed_experiments: 2,
    duration_s: 20,
    cost_usd: 0,
    exhausted: [],
    can_schedule: true,
  },
  counts: {
    experiments: 2,
    measured: 2,
    promoted: 1,
    parked: 1,
    rejected: 0,
    crashed: 0,
  },
  incumbent: {
    commit: '1234567890abcdef1234567890abcdef12345678',
    source: 'promoted_experiment',
    experiment_id: experiment.experiment_id,
    recorded_at: experiment.recorded_at,
    experiment,
  },
  latest_measured: experiment,
  active_beam: [],
  frontier: [experiment],
  experiments: [experiment],
};

describe('CampaignSnapshot', () => {
  it('shows incumbent lineage, confidence bounds, frozen cases, and profile links', () => {
    render(<CampaignSnapshot data={data} />);

    expect(screen.getAllByText('exp-002').length).toBeGreaterThan(0);
    expect(screen.getAllByText('+5.00% to +11.00%').length).toBeGreaterThan(0);
    expect(screen.getByText('Frozen')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /candidate\.ncu-rep/i })).toHaveAttribute(
      'href',
      expect.stringContaining('/api/optimization/campaign/artifact')
    );
  });

  it('shows the frozen control commit before the first promotion', () => {
    render(
      <CampaignSnapshot
        data={{
          ...data,
          incumbent: {
            commit: 'abcdef1234567890abcdef1234567890abcdef12',
            source: 'initial_control',
            experiment_id: null,
            recorded_at: '2026-08-16T00:00:00+00:00',
            experiment: null,
          },
        }}
      />
    );

    expect(screen.getByText('Frozen control')).toBeInTheDocument();
    expect(screen.getByText('abcdef123456')).toBeInTheDocument();
  });
});
