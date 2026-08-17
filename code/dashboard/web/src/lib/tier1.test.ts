import { isTier1ReleaseReady, tier1CanonicalRunCount } from '@/lib/tier1';
import type { Tier1History } from '@/types';

function readyHistory(): Tier1History {
  const run = {
    run_id: 'run-ready',
    run_accepted: true,
    baseline_eligible: true,
    target_count: 2,
    succeeded: 2,
    failed: 0,
    skipped: 0,
    missing: 0,
    avg_speedup: 2,
    median_speedup: 2,
    geomean_speedup: 2,
    representative_speedup: 2,
    max_speedup: 2,
  };

  return {
    suite_name: 'tier1',
    total_runs: 1,
    latest_run_id: run.run_id,
    runs: [run],
    latest: {
      run,
      summary: {},
      targets: [],
      regressions: [],
      improvements: [],
      new_targets: [],
      missing_targets: [],
      anchor_declines: [],
      suppressed_regressions: [],
      rechecks: [],
    },
    latest_accepted: {
      run,
      summary: {},
      targets: [],
      regressions: [],
      improvements: [],
      new_targets: [],
      missing_targets: [],
      anchor_declines: [],
      suppressed_regressions: [],
      rechecks: [],
    },
    warnings: [],
  };
}

const blockers: Array<[string, (history: Tier1History) => void]> = [
  ['an unaccepted run', (history) => (history.latest.run!.run_accepted = false)],
  [
    'an empty run',
    (history) => {
      history.latest.run!.target_count = 0;
      history.latest.run!.succeeded = 0;
    },
  ],
  ['an incomplete run', (history) => (history.latest.run!.succeeded = 1)],
  ['a failed target', (history) => (history.latest.run!.failed = 1)],
  ['a skipped target', (history) => (history.latest.run!.skipped = 1)],
  ['a missing target count', (history) => (history.latest.run!.missing = 1)],
  [
    'a regression',
    (history) => history.latest.regressions.push({ target: 'ch01:demo', reason: 'slower' }),
  ],
  [
    'a deleted target',
    (history) => history.latest.missing_targets.push({ target: 'ch01:deleted' }),
  ],
  [
    'a recheck-suppressed regression',
    (history) =>
      history.latest.suppressed_regressions.push({
        target: 'ch01:demo',
        reason: 'recheck_not_regressed',
      }),
  ],
  ['a history warning', (history) => history.warnings.push('history index is incomplete')],
];

describe('isTier1ReleaseReady', () => {
  it('accepts a complete eligible run with no regressions or history warnings', () => {
    expect(isTier1ReleaseReady(readyHistory())).toBe(true);
  });

  it('rejects missing history and a missing latest run', () => {
    expect(isTier1ReleaseReady(null)).toBe(false);

    const history = readyHistory();
    history.latest.run = null;
    expect(isTier1ReleaseReady(history)).toBe(false);
  });

  it.each(blockers)('rejects %s', (_description, block) => {
    const history = readyHistory();
    block(history);

    expect(isTier1ReleaseReady(history)).toBe(false);
  });

  it('accepts a healthy run with a small anchor decline while the prior baseline remains', () => {
    const history = readyHistory();
    history.latest.run!.baseline_eligible = false;
    history.latest.anchor_declines.push({
      target: 'ch01:demo',
      reason: 'optimized_latency',
      delta_pct: 2,
    });

    expect(isTier1ReleaseReady(history)).toBe(true);
  });
});

describe('tier1CanonicalRunCount', () => {
  it('counts accepted history without labeling rejected evidence as canonical', () => {
    const history = readyHistory();
    history.total_runs = 2;
    history.accepted_runs = 1;
    history.runs.push({
      ...history.runs[0],
      run_id: 'rejected-evidence',
      run_accepted: false,
      baseline_eligible: false,
    });

    expect(tier1CanonicalRunCount(history)).toBe(1);
  });
});
