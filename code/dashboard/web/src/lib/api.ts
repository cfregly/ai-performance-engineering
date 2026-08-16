/**
 * API Client - MCP/Engine aligned endpoints only.
 */

import type {
  BenchmarkContractsSummary,
  BenchmarkE2EStatusSnapshot,
  BenchmarkRunGeneratorDefaults,
  BenchmarkRunRenderResult,
  CampaignDashboardResult,
} from '@/types';

const API_BASE = '/api';

export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

type Envelope<T> = {
  status?: 'ok' | 'error';
  result?: T;
  error?: string;
  success?: boolean;
};

function unwrapEnvelope<T>(payload: unknown): T {
  if (payload && typeof payload === 'object' && 'result' in payload) {
    return (payload as Envelope<T>).result as T;
  }
  return payload as T;
}

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit,
  opts?: { allowErrorResult?: boolean }
): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    throw new APIError(res.status, `API error: ${res.status} ${res.statusText}`);
  }

  const payload = await res.json();
  const envelope = payload as Envelope<T>;

  if (envelope.status === 'error' && !opts?.allowErrorResult) {
    const message =
      (typeof envelope.result === 'object' && envelope.result && 'error' in envelope.result
        ? (envelope.result as any).error
        : envelope.error) || 'API error';
    throw new APIError(res.status, message);
  }

  return unwrapEnvelope<T>(payload);
}

function buildQuery(params?: Record<string, unknown>): string {
  if (!params) return '';
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') {
      return;
    }
    if (Array.isArray(value)) {
      const joined = value
        .filter((item) => item !== undefined && item !== null && String(item).trim() !== '')
        .map((item) => String(item))
        .join(',');
      if (joined) {
        search.set(key, joined);
      }
    } else {
      search.set(key, String(value));
    }
  });
  const query = search.toString();
  return query ? `?${query}` : '';
}

// ============================================================================
// CORE DATA ENDPOINTS
// ============================================================================

export async function getBenchmarkData(params?: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI(`/benchmark/data${query}`);
}

export async function getBenchmarkOverview() {
  return fetchAPI('/benchmark/overview');
}

export async function getBenchmarkContracts() {
  return fetchAPI<BenchmarkContractsSummary>('/benchmark/contracts');
}

export async function renderBenchmarkRun(
  params: Partial<BenchmarkRunGeneratorDefaults>
) {
  return fetchAPI<BenchmarkRunRenderResult>('/benchmark/contracts/render-run', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getBenchmarkHistory() {
  return fetchAPI('/benchmark/history');
}

export async function getBenchmarkTrends() {
  return fetchAPI('/benchmark/trends');
}

export async function getBenchmarkE2eStatus(params?: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI<BenchmarkE2EStatusSnapshot>(`/benchmark/e2e-status${query}`, undefined, { allowErrorResult: true });
}

export async function getTier1History() {
  return fetchAPI('/benchmark/tier1/history');
}

export async function getTier1Trends() {
  return fetchAPI('/benchmark/tier1/trends');
}

export async function getTier1TargetHistory(params: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI(`/benchmark/tier1/target-history${query}`);
}

export async function getBenchmarkCompare(params: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI(`/benchmark/compare${query}`);
}

export async function getOptimizationCampaign(workspace: string) {
  const query = buildQuery({ workspace });
  return fetchAPI<CampaignDashboardResult>(`/optimization/campaign${query}`);
}

export async function getGpuInfo() {
  return fetchAPI('/gpu/info');
}

export async function getSoftwareInfo() {
  return fetchAPI('/system/software');
}

export async function getDependencies() {
  return fetchAPI('/system/dependencies');
}

export async function runClockLockCheck(params?: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI(`/system/clock-lock-check${query}`, undefined, { allowErrorResult: true });
}

// ============================================================================
// PROFILING
// ============================================================================

export async function getProfilePairs() {
  return fetchAPI('/profile/list');
}

export async function getCompileAnalysis() {
  return fetchAPI('/profile/compile');
}

export async function getNcuSummary(params: Record<string, unknown>) {
  const query = buildQuery(params);
  return fetchAPI(`/profile/ncu-summary${query}`, undefined, { allowErrorResult: true });
}

// ============================================================================
// CLUSTER
// ============================================================================

export async function runClusterEvalSuite(params: Record<string, unknown>) {
  return fetchAPI(
    '/cluster/eval-suite',
    {
      method: 'POST',
      body: JSON.stringify(params),
    },
    { allowErrorResult: true }
  );
}

export async function runClusterCommonEval(params: Record<string, unknown>) {
  return fetchAPI(
    '/cluster/common-eval',
    {
      method: 'POST',
      body: JSON.stringify(params),
    },
    { allowErrorResult: true }
  );
}

export async function buildCanonicalPackage(params: Record<string, unknown>) {
  return fetchAPI(
    '/cluster/build-canonical-package',
    {
      method: 'POST',
      body: JSON.stringify(params),
    },
    { allowErrorResult: true }
  );
}

export async function promoteClusterRun(params: Record<string, unknown>) {
  return fetchAPI(
    '/cluster/promote-run',
    {
      method: 'POST',
      body: JSON.stringify(params),
    },
    { allowErrorResult: true }
  );
}

export async function validateFieldReport(params: Record<string, unknown>) {
  return fetchAPI(
    '/cluster/validate-field-report',
    {
      method: 'POST',
      body: JSON.stringify(params),
    },
    { allowErrorResult: true }
  );
}

// ============================================================================
// AI / TOOLING
// ============================================================================

export async function getAiStatus() {
  return fetchAPI('/ai/status');
}

export async function getAiTools() {
  return fetchAPI('/ai/tools');
}

export async function executeAiTool(tool: string, params: Record<string, unknown>) {
  return fetchAPI(
    '/ai/execute',
    {
      method: 'POST',
      body: JSON.stringify({ tool, params }),
    },
    { allowErrorResult: true }
  );
}
