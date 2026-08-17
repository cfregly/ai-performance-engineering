import type { Tier1History } from '@/types';

export function tier1CanonicalRunCount(history: Tier1History | null): number {
  if (!history) {
    return 0;
  }
  return (
    history.accepted_runs ?? history.runs.filter((run) => run.run_accepted === true).length
  );
}

export function isTier1ReleaseReady(history: Tier1History | null): boolean {
  const latestRun = history?.latest?.run;
  if (!latestRun || latestRun.run_accepted !== true) {
    return false;
  }

  return (
    latestRun.target_count > 0 &&
    latestRun.succeeded === latestRun.target_count &&
    latestRun.failed === 0 &&
    latestRun.skipped === 0 &&
    latestRun.missing === 0 &&
    (history.latest.regressions?.length ?? 0) === 0 &&
    (history.latest.suppressed_regressions?.length ?? 0) === 0 &&
    (history.latest.missing_targets?.length ?? 0) === 0 &&
    (history.warnings?.length ?? 0) === 0
  );
}
