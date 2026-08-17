export interface Benchmark {
  name: string;
  chapter: string;
  type: string;
  status: 'succeeded' | 'failed' | 'skipped';
  baseline_time_ms: number;
  optimized_time_ms: number;
  speedup: number;
  raw_speedup?: number;
  speedup_capped?: boolean;
  optimization_goal?: 'speed' | 'memory';
  baseline_memory_mb?: number | null;
  optimized_memory_mb?: number | null;
  memory_savings_pct?: number | null;
  p75_ms?: number | null;
  error?: string;
}

export interface BenchmarkSummary {
  total: number;
  succeeded: number;
  failed: number;
  skipped: number;
  avg_speedup: number;
  max_speedup: number;
  min_speedup: number;
}

export interface BenchmarkPagination {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface BenchmarkFilters {
  search?: string | null;
  status?: string[];
  chapter?: string[];
  benchmark?: string | null;
  optimization_goal?: string | null;
  sort_field?: string;
  sort_dir?: string;
}

export interface BenchmarkOverview {
  timestamp?: string;
  summary: BenchmarkSummary;
  status_counts: {
    succeeded: number;
    failed: number;
    skipped: number;
  };
  top_speedups: Benchmark[];
  chapter_stats: Array<{
    chapter: string;
    count: number;
    succeeded: number;
    avg_speedup: number;
    max_speedup: number;
  }>;
}

export interface BenchmarkContractYamlSummary {
  top_level_keys: string[];
  spec_keys?: string[];
  enabled_layers?: string[];
  has_observability?: boolean;
  has_sinks?: boolean;
}

export interface BenchmarkContractEntry {
  name: string;
  path: string;
  exists: boolean;
  kind: 'doc' | 'yaml';
  description: string;
  summary?: BenchmarkContractYamlSummary;
}

export interface BenchmarkContractInterfaceEntry {
  id: 'cli' | 'dashboard_api' | 'mcp';
  label: string;
  transport: 'cli' | 'http' | 'mcp';
  entrypoint: string;
  method?: string;
  description: string;
}

export interface BenchmarkRunGeneratorDefaults {
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
}

export interface BenchmarkRunRenderResult {
  schema_version: string;
  generated_at_utc: string;
  template_path: string;
  applied_values: BenchmarkRunGeneratorDefaults;
  rendered_yaml: string;
}

export interface BenchmarkContractsSummary {
  schema_version: string;
  generated_at_utc: string;
  available: boolean;
  repo_root: string;
  surface_order: string[];
  surface_count: number;
  missing_surface_count: number;
  missing_surfaces: string[];
  interfaces: {
    cli: string;
    dashboard_api: string;
    mcp_tool: string;
  };
  interface_entries: BenchmarkContractInterfaceEntry[];
  generator: {
    template_path: string;
    defaults: BenchmarkRunGeneratorDefaults;
    preview_yaml: string;
    choices: {
      benchmarkClass: BenchmarkRunGeneratorDefaults['benchmarkClass'][];
      workloadType: BenchmarkRunGeneratorDefaults['workloadType'][];
      cadence: BenchmarkRunGeneratorDefaults['cadence'][];
      comparisonVariable: BenchmarkRunGeneratorDefaults['comparisonVariable'][];
    };
    render_interfaces: {
      cli: string;
      dashboard_api: string;
      mcp_tool: string;
    };
    render_interface_entries: BenchmarkContractInterfaceEntry[];
  };
  contracts: Record<string, BenchmarkContractEntry>;
}

export interface BenchmarkPage {
  timestamp?: string;
  summary: BenchmarkSummary;
  benchmarks: Benchmark[];
  pagination: BenchmarkPagination;
  filters: BenchmarkFilters;
}

export interface BenchmarkData {
  benchmarks: Benchmark[];
  summary: BenchmarkSummary;
  timestamp: string;
  speedup_cap?: number;
}

export interface GpuInfo {
  name: string;
  memory_total: number;      // MB
  memory_used: number;       // MB
  utilization: number;
  temperature: number;
  temperature_hbm?: number | null;
  power?: number;            // watts
  power_draw?: number;       // legacy key
  power_limit?: number;      // watts
  compute_capability?: string;
  driver_version?: string;
  cuda_version?: string;
  clock_graphics?: number;
  clock_memory?: number;
  fan_speed?: number | null;
  pstate?: string | null;
  live?: boolean;
}

export interface SoftwareInfo {
  python_version: string;
  pytorch_version: string;
  cuda_version: string;
  cudnn_version: string;
  triton_version?: string;
}

export interface LLMAnalysis {
  summary: string;
  key_findings: string[];
  recommendations: string[];
  bottlenecks: Array<{
    name: string;
    severity: 'high' | 'medium' | 'low';
    description: string;
    recommendation: string;
  }>;
}

export interface ProfilerData {
  kernels: Array<{
    name: string;
    duration_ms: number;
    memory_mb: number;
    occupancy: number;
  }>;
  memory_timeline: Array<{
    timestamp: number;
    allocated_mb: number;
    reserved_mb: number;
  }>;
  flame_data?: unknown;
}

export interface ProfilePair {
  chapter: string;
  name: string;
  path: string;
  run_id: string;
  type?: string;
  has_nsys: boolean;
  has_ncu: boolean;
  baseline_nsys: string[];
  optimized_nsys: string[];
  baseline_ncu: string[];
  optimized_ncu: string[];
}

export interface ProfilePairsResult {
  pairs: ProfilePair[];
  count: number;
}

export interface CompileAnalysisBenchmark {
  name: string;
  chapter: string;
  speedup: number;
  baseline_time_ms: number;
  optimized_time_ms: number;
}

export interface CompileAnalysis {
  speedup: number;
  compile_time_ms: number;
  graph_breaks: number;
  fusion_ratio: number;
  recommendations: string[];
  compile_benchmarks: CompileAnalysisBenchmark[];
  has_real_data: boolean;
  mode_comparison?: Record<string, unknown>;
  graph_breaks_list?: unknown[];
}

export interface BenchmarkE2EStatusStage {
  name: string;
  status: string;
  description?: string;
  run_id?: string;
  issues?: string[];
  attempts?: Array<Record<string, unknown>>;
  failed_benchmarks?: Array<Record<string, unknown>>;
}

export interface BenchmarkE2EIssueGroup {
  group_id: string;
  stage: string;
  bucket?: string | null;
  benchmark_status?: string | null;
  signature: string;
  signature_key?: string | null;
  root_cause_hint?: string | null;
  count: number;
  sample_targets?: string[];
  first_reported_at?: string | null;
  last_reported_at?: string | null;
  cascade_from?: string | null;
}

export interface BenchmarkE2EStatusSnapshot {
  success: boolean;
  error?: string;
  run_id?: string;
  run_dir?: string;
  run_state?: string;
  overall_status?: string;
  stored_overall_status?: string;
  inferred_state?: string;
  resume_available?: boolean;
  stored_resume_available?: boolean;
  notes?: string[];
  progress_source?: {
    kind?: string;
    label?: string;
    reason?: string;
    progress_timestamp?: string | null;
    progress_path?: string | null;
    summary_progress_lag_seconds?: number | null;
    checkpoint_progress_lag_seconds?: number | null;
    status_command_shell?: string;
    status_api_path?: string;
    dashboard_path?: string;
    preferred_mcp_tool?: string;
  };
  current?: {
    stage?: string | null;
    bucket?: string | null;
    child_run_id?: string | null;
    step?: string | null;
    detail?: string | null;
    percent_complete?: number | null;
    elapsed_seconds?: number | null;
    eta_seconds?: number | null;
    child_progress?: Record<string, unknown> | null;
    child_artifacts?: Record<string, string | null> | null;
    recent_child_events?: Array<Record<string, unknown>>;
    reported_failures?: Array<Record<string, unknown>>;
  };
  liveness?: {
    orchestrator_pid?: number | null;
    orchestrator_live?: boolean;
    watcher_pid?: number | null;
    watcher_live?: boolean;
  };
  watcher?: {
    watcher_pid?: number | null;
    watch_state?: string | null;
    stored_watch_state?: string | null;
    watcher_live?: boolean;
    poll_interval_seconds?: number | null;
    max_auto_resumes?: number | null;
    auto_resume_count?: number | null;
    last_action?: Record<string, unknown> | null;
  } | null;
  stages?: BenchmarkE2EStatusStage[];
  issue_groups?: BenchmarkE2EIssueGroup[];
  historical_issue_groups?: BenchmarkE2EIssueGroup[];
  recent_events?: Array<Record<string, unknown>>;
  ledgers?: {
    summary?: {
      issue_count?: number;
      issue_group_count?: number;
      reported_issue_count?: number;
      resolved_count?: number;
      unresolved_count?: number;
      active_issue_count?: number;
      active_unresolved_count?: number;
      active_reported_issue_count?: number;
      active_issue_group_count?: number;
      historical_issue_count?: number;
      historical_unresolved_count?: number;
      historical_reported_issue_count?: number;
      historical_issue_group_count?: number;
      run_id?: string;
    };
    active_issue_ledger_json?: string | null;
    active_issue_ledger_md?: string | null;
    historical_failure_ledger_json?: string | null;
    historical_failure_ledger_md?: string | null;
  };
  actions?: {
    status_command_shell?: string;
    watch_command_shell?: string;
    watcher_command_shell?: string;
    resume_command_shell?: string;
    status_api_path?: string;
    watcher_api_path?: string;
    dashboard_path?: string;
    preferred_mcp_tool?: string;
    watcher_mcp_tool?: string;
  };
  paths?: Record<string, string>;
}

export interface NcuKernelRow {
  id: number;
  kernel_name: string;
  block_size: string;
  grid_size: string;
  stream: string;
  device: string;
  cc: string;
  time_avg_ms?: number | null;
  time_sum_ms?: number | null;
  time_pct?: number | null;
  occupancy_limit_reason?: string | null;
  metrics: Record<string, number>;
}

export interface NcuSummaryResult {
  success: boolean;
  report_path?: string;
  top_k?: number;
  sort_by?: string;
  kernel_count?: number;
  total_time_sum_ms?: number | null;
  kernels?: NcuKernelRow[];
  metrics_requested?: string[];
  command?: string[] | null;
  stderr?: string | null;
  returncode?: number | null;
  error?: string;
}

export interface ClockLockSnapshot {
  app_sm_mhz?: number | null;
  app_mem_mhz?: number | null;
  cur_sm_mhz?: number | null;
  cur_mem_mhz?: number | null;
  error?: string;
}

export interface ClockLockResultRow {
  device: number;
  physical_index: number;
  locked: boolean;
  error?: string;
  physical_index_error?: string;
  theoretical_tflops_fp16?: number;
  theoretical_gbps?: number;
  before?: ClockLockSnapshot;
  during?: ClockLockSnapshot;
  after?: ClockLockSnapshot;
}

export interface ClockLockCheckResult {
  success: boolean;
  gpu_count?: number;
  results?: ClockLockResultRow[];
  error?: string;
}

export interface CommandRunResult {
  command: string[];
  returncode: number | null;
  stdout: string;
  stderr: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterEvalSuiteResult {
  success: boolean;
  mode?: string;
  run_id?: string;
  run_dir?: string;
  structured_dir?: string;
  raw_dir?: string;
  figures_dir?: string;
  reports_dir?: string;
  primary_label?: string;
  meta_path?: string;
  manifest_path?: string | null;
  collect?: CommandRunResult;
  manifest?: CommandRunResult;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterCommonEvalResult extends ClusterEvalSuiteResult {
  preset?: string;
  preset_description?: string;
  artifact_roles?: string[];
  coverage_baseline_run_id?: string | null;
}

export interface ClusterCanonicalPackageResult {
  success: boolean;
  canonical_run_id?: string;
  comparison_run_ids?: string[];
  historical_run_ids?: string[];
  output_dir?: string;
  package_readme_path?: string;
  package_manifest_path?: string;
  cleanup_keep_run_ids_path?: string;
  historical_reference_path?: string;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterPromoteRunResult {
  success: boolean;
  run_id?: string;
  label?: string;
  run_dir?: string;
  published_root?: string;
  published_structured_dir?: string;
  published_raw_dir?: string;
  published_figures_dir?: string;
  published_reports_dir?: string;
  published_manifest_path?: string;
  published_localhost_report_path?: string;
  published_localhost_notes_path?: string;
  allow_run_ids?: string[];
  steps?: Record<string, unknown>;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface FieldReportValidationResult {
  success: boolean;
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface BenchmarkRunSummary {
  date: string;
  timestamp: string;
  benchmark_count: number;
  avg_speedup: number;
  max_speedup: number;
  successful: number;
  failed: number;
  source: string;
}

export interface BenchmarkHistory {
  total_runs: number;
  latest?: string | null;
  runs: BenchmarkRunSummary[];
}

export interface BenchmarkTrendPoint {
  date: string;
  avg_speedup: number;
  max_speedup: number;
  benchmark_count: number;
}

export interface BenchmarkTrends {
  history: BenchmarkTrendPoint[];
  by_date: BenchmarkTrendPoint[];
  avg_speedup: number;
  run_count: number;
  best_ever?: {
    date?: string;
    speedup?: number;
  };
  improvements?: Array<{
    date: string;
    delta: number;
  }>;
  regressions?: Array<{
    date: string;
    delta: number;
  }>;
}

export interface Tier1RunSummary {
  run_id: string;
  run_accepted: boolean;
  baseline_eligible: boolean;
  generated_at?: string | null;
  suite_name?: string;
  suite_version?: number;
  target_count: number;
  succeeded: number;
  failed: number;
  skipped: number;
  missing: number;
  avg_speedup: number;
  median_speedup: number;
  geomean_speedup: number;
  representative_speedup: number;
  max_speedup: number;
  summary_path?: string | null;
  regression_summary_path?: string | null;
  regression_summary_json_path?: string | null;
  trend_snapshot_path?: string | null;
  source_result_json?: string | null;
  improvement_count?: number;
  regression_count?: number;
}

export interface Tier1TargetSummary {
  key: string;
  target: string;
  category: string;
  rationale?: string;
  status: string;
  baseline_time_ms?: number | null;
  best_speedup?: number | null;
  best_optimized_time_ms?: number | null;
  best_optimization?: string | null;
  optimization_goal?: string | null;
  baseline_memory_mb?: number | null;
  best_memory_savings_pct?: number | null;
  best_optimized_memory_mb?: number | null;
  baseline_p75_ms?: number | null;
  baseline_file?: string | null;
  artifacts?: Record<string, string>;
}

export interface Tier1LatestRunDetails {
  run: Tier1RunSummary | null;
  summary: Record<string, unknown>;
  targets: Tier1TargetSummary[];
  regressions: Tier1Delta[];
  improvements: Tier1Delta[];
  new_targets: Tier1Delta[];
  missing_targets: Tier1Delta[];
  anchor_declines: Tier1Delta[];
  suppressed_regressions: Tier1Delta[];
  rechecks: Record<string, unknown>[];
}

export interface Tier1Delta {
  key?: string;
  target?: string;
  reason?: string;
  before?: string | number | null;
  after?: string | number | null;
  delta?: number | null;
  delta_pct?: number | null;
}

export interface Tier1History {
  suite_name: string;
  suite_version?: number;
  history_root?: string;
  total_runs: number;
  accepted_runs?: number;
  latest_evidence_run_id?: string | null;
  latest_run_id?: string | null;
  runs: Tier1RunSummary[];
  latest: Tier1LatestRunDetails;
  latest_accepted: Tier1LatestRunDetails;
  warnings: string[];
}

export interface Tier1TrendPoint {
  run_id: string;
  generated_at: string;
  avg_speedup: number;
  median_speedup: number;
  geomean_speedup: number;
  representative_speedup: number;
  max_speedup: number;
  succeeded: number;
  failed: number;
  skipped: number;
  missing: number;
}

export interface Tier1Trends {
  suite_name: string;
  run_count: number;
  latest_run_id?: string | null;
  history: Tier1TrendPoint[];
  avg_speedup: number;
  avg_median_speedup?: number;
  avg_geomean_speedup?: number;
  representative_speedup?: number;
  best_speedup_seen?: number;
}

export interface Tier1TargetHistoryPoint {
  run_id: string;
  run_accepted?: boolean;
  baseline_eligible?: boolean;
  generated_at?: string | null;
  key?: string | null;
  target?: string | null;
  category?: string | null;
  status: string;
  baseline_time_ms?: number | null;
  best_optimized_time_ms?: number | null;
  best_speedup?: number | null;
  best_optimization?: string | null;
  baseline_memory_mb?: number | null;
  best_memory_savings_pct?: number | null;
  artifacts?: Record<string, string>;
}

export interface Tier1TargetHistory {
  suite_name: string;
  suite_version?: number;
  history_root?: string;
  selected_key?: string | null;
  selected_target?: string | null;
  category?: string | null;
  rationale?: string | null;
  run_count: number;
  evidence_run_count?: number;
  best_speedup_seen?: number;
  latest?: Tier1TargetHistoryPoint | null;
  latest_evidence?: Tier1TargetHistoryPoint | null;
  history: Tier1TargetHistoryPoint[];
  evidence_history?: Tier1TargetHistoryPoint[];
}

export interface BenchmarkCompareDelta {
  key: string;
  chapter: string;
  name: string;
  baseline_speedup: number;
  candidate_speedup: number;
  delta: number;
  delta_pct?: number | null;
  baseline_status: string;
  candidate_status: string;
  status_changed: boolean;
  baseline_time_ms?: number | null;
  candidate_time_ms?: number | null;
  baseline_optimized_time_ms?: number | null;
  candidate_optimized_time_ms?: number | null;
}

export interface BenchmarkCompareRun {
  path: string;
  timestamp?: string | null;
  summary: BenchmarkSummary;
}

export interface BenchmarkCompareBenchmark {
  key: string;
  chapter: string;
  name: string;
  status: string;
  speedup: number;
  baseline_time_ms?: number | null;
  optimized_time_ms?: number | null;
  optimization_goal?: string | null;
}

export interface BenchmarkCompareResult {
  baseline: BenchmarkCompareRun;
  candidate: BenchmarkCompareRun;
  overlap: {
    common: number;
    added: number;
    removed: number;
    baseline_total: number;
    candidate_total: number;
  };
  deltas: BenchmarkCompareDelta[];
  regressions: BenchmarkCompareDelta[];
  improvements: BenchmarkCompareDelta[];
  added_benchmarks: BenchmarkCompareBenchmark[];
  removed_benchmarks: BenchmarkCompareBenchmark[];
  status_transitions: Record<string, number>;
}

export interface CampaignGateDecision {
  decision: 'promote' | 'park' | 'reject' | 'inconclusive' | string;
  reasons: string[];
  aggregate_control?: number | null;
  aggregate_candidate?: number | null;
  improvement_pct?: number | null;
  improvement_ci_pct?: number[] | null;
  case_improvements_pct?: Record<string, number>;
  case_improvement_ci_pct?: Record<string, number[]>;
  confidence_method?: string | null;
  confidence_level?: number | null;
  confidence_required?: boolean;
  minimum_trials?: number;
}

export interface CampaignCaseRow {
  case_id: string;
  primary: boolean;
  frozen: boolean;
  control_median: number;
  candidate_median: number;
  improvement_pct: number;
  improvement_ci_pct?: number[] | null;
  control_trials: number;
  candidate_trials: number;
  frozen_case_violation: boolean;
}

export interface CampaignArtifact {
  role: 'raw' | 'profile' | string;
  path: string;
  sha256?: string | null;
  downloadable: boolean;
}

export interface CampaignExperimentSummary {
  experiment_id: string;
  parent_id?: string | null;
  beam: string;
  hypothesis: string;
  status: string;
  correctness: string;
  primary_case: string;
  changed_surface: string[];
  mechanism: string;
  outcome: string;
  next_step: string;
  recorded_at: string;
  revision: number;
  is_incumbent: boolean;
  gate: CampaignGateDecision;
  cases: CampaignCaseRow[];
  artifacts: CampaignArtifact[];
  provenance: Record<string, unknown>;
}

export interface CampaignIncumbent {
  commit: string;
  source: 'initial_control' | 'promoted_experiment' | string;
  experiment_id?: string | null;
  recorded_at: string;
  experiment?: CampaignExperimentSummary | null;
}

export interface CampaignDashboardResult {
  workspace: string;
  config: Record<string, any>;
  budget: {
    completed_experiments: number;
    duration_s: number;
    cost_usd: number;
    exhausted: string[];
    can_schedule: boolean;
    [key: string]: unknown;
  };
  counts: {
    experiments: number;
    measured: number;
    promoted: number;
    parked: number;
    rejected: number;
    crashed: number;
  };
  incumbent: CampaignIncumbent;
  latest_measured?: CampaignExperimentSummary | null;
  active_beam: CampaignExperimentSummary[];
  frontier: CampaignExperimentSummary[];
  experiments: CampaignExperimentSummary[];
}

export interface Tab {
  id: string;
  label: string;
  icon: string;
}
