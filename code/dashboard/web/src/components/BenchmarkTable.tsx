'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Pin, Star, Target, FileCode2, X } from 'lucide-react';
import { Benchmark, BenchmarkPagination } from '@/types';
import { formatMs, getSpeedupColor, cn } from '@/lib/utils';
import { getBenchmarkData } from '@/lib/api';

interface BenchmarkTableProps {
  chapters?: string[];
  externalFilter?: { type: 'benchmark' | 'chapter'; value: string };
  onClearExternalFilter?: () => void;
  refreshKey?: number;
  pinnedBenchmarks?: Set<string>;
  favorites?: Set<string>;
  speedupCap?: number;
  onTogglePin?: (key: string) => void;
  onToggleFavorite?: (key: string) => void;
  onFocusBenchmark?: (benchmark: Benchmark) => void;
  onShowCodeDiff?: (benchmark: Benchmark) => void;
}

type SortField = 'name' | 'chapter' | 'speedup' | 'baseline_time_ms' | 'optimized_time_ms' | 'status';
type SortDir = 'asc' | 'desc';

const DEFAULT_PAGE_SIZE = 25;

export function BenchmarkTable({
  chapters = [],
  externalFilter,
  onClearExternalFilter,
  refreshKey,
  pinnedBenchmarks = new Set(),
  favorites = new Set(),
  speedupCap,
  onTogglePin,
  onToggleFavorite,
  onFocusBenchmark,
  onShowCodeDiff,
}: BenchmarkTableProps) {
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [sortField, setSortField] = useState<SortField>('speedup');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [chapterFilter, setChapterFilter] = useState<string>('all');
  const [benchmarkFilter, setBenchmarkFilter] = useState<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [pagination, setPagination] = useState<BenchmarkPagination | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlightKey, setHighlightKey] = useState<string | null>(null);

  useEffect(() => {
    const handle = window.setTimeout(() => setDebouncedSearch(search), 250);
    return () => window.clearTimeout(handle);
  }, [search]);

  useEffect(() => {
    if (!externalFilter) return;
    if (externalFilter.type === 'benchmark') {
      setBenchmarkFilter(externalFilter.value);
      setSearch(externalFilter.value);
      setChapterFilter('all');
      setHighlightKey(externalFilter.value);
    }
    if (externalFilter.type === 'chapter') {
      setBenchmarkFilter(null);
      setSearch('');
      setChapterFilter(externalFilter.value);
      setHighlightKey(null);
    }
    setPage(1);
  }, [externalFilter]);

  useEffect(() => {
    if (!externalFilter) {
      setHighlightKey(null);
      setBenchmarkFilter(null);
    }
  }, [externalFilter]);

  const fetchBenchmarks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getBenchmarkData({
        page,
        page_size: pageSize,
        search: debouncedSearch || undefined,
        status: statusFilter === 'all' ? undefined : statusFilter,
        chapter: chapterFilter === 'all' ? undefined : chapterFilter,
        benchmark: benchmarkFilter || undefined,
        sort_field: sortField,
        sort_dir: sortDir,
      });
      const result = data as any;
      setBenchmarks(result.benchmarks || []);
      setPagination(result.pagination || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load benchmarks');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, debouncedSearch, statusFilter, chapterFilter, benchmarkFilter, sortField, sortDir]);

  useEffect(() => {
    fetchBenchmarks();
  }, [fetchBenchmarks, refreshKey]);

  const chapterOptions = useMemo(() => {
    if (chapters.length) return chapters;
    return Array.from(new Set(benchmarks.map((b) => b.chapter))).sort();
  }, [benchmarks, chapters]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
    setPage(1);
  };

  const getSortIcon = (field: SortField) => {
    if (field !== sortField) return <ArrowUpDown className="w-4 h-4 opacity-30" />;
    return sortDir === 'asc' ? (
      <ArrowUp className="w-4 h-4 text-accent-primary" />
    ) : (
      <ArrowDown className="w-4 h-4 text-accent-primary" />
    );
  };

  const getBenchmarkKey = (b: Benchmark) => `${b.chapter}:${b.name}`;

  const total = pagination?.total ?? 0;
  const totalPages = Math.max(1, pagination?.total_pages ?? 1);
  const startIndex = total ? (page - 1) * pageSize + 1 : 0;
  const endIndex = total ? startIndex + benchmarks.length - 1 : 0;
  const columnCount = (onTogglePin || onToggleFavorite) ? 8 : 7;

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);

  const clearExternalFilter = () => {
    setBenchmarkFilter(null);
    setHighlightKey(null);
    if (externalFilter?.type === 'chapter') {
      setChapterFilter('all');
    }
    if (externalFilter?.type === 'benchmark') {
      setSearch('');
    }
    onClearExternalFilter?.();
  };

  return (
    <div id="benchmark-table" className="card">
      <div className="card-header flex-col sm:flex-row gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-white">Benchmarks</h2>
          {externalFilter && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-accent-primary/10 border border-accent-primary/30 rounded-full text-xs text-accent-primary">
              <span>
                {externalFilter.type === 'chapter'
                  ? `Chapter ${externalFilter.value}`
                  : `Benchmark ${externalFilter.value}`}
              </span>
              <button onClick={clearExternalFilter} className="text-accent-primary/80 hover:text-accent-primary">
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {pinnedBenchmarks.size > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-accent-primary/10 border border-accent-primary/30 rounded-full">
              <Pin className="w-4 h-4 text-accent-primary" />
              <span className="text-sm text-accent-primary">{pinnedBenchmarks.size} pinned</span>
            </div>
          )}

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              id="benchmark-search"
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setBenchmarkFilter(null);
                if (externalFilter) {
                  onClearExternalFilter?.();
                }
                setPage(1);
              }}
              className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/40 focus:outline-hidden focus:border-accent-primary/50 w-48"
            />
          </div>

          <select
            value={chapterFilter}
            onChange={(e) => {
              setChapterFilter(e.target.value);
              setBenchmarkFilter(null);
              if (externalFilter) {
                onClearExternalFilter?.();
              }
              setPage(1);
            }}
            className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
          >
            <option value="all">All Chapters</option>
            {chapterOptions.map((ch) => (
              <option key={ch} value={ch}>
                {ch}
              </option>
            ))}
          </select>

          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setBenchmarkFilter(null);
              if (externalFilter) {
                onClearExternalFilter?.();
              }
              setPage(1);
            }}
            className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
          >
            <option value="all">All Status</option>
            <option value="succeeded">Succeeded</option>
            <option value="failed">Failed</option>
            <option value="skipped">Skipped</option>
          </select>

          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-hidden"
          >
            {[10, 25, 50, 100].map((size) => (
              <option key={size} value={size}>
                {size} / page
              </option>
            ))}
          </select>

          <span className="text-sm text-white/40">
            {loading ? 'Loading...' : `${total} results`}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/5">
              {(onTogglePin || onToggleFavorite) && <th className="px-3 py-3 w-16"></th>}
              <th
                className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('name')}
              >
                <div className="flex items-center gap-2">Name {getSortIcon('name')}</div>
              </th>
              <th
                className="px-5 py-3 text-left text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('chapter')}
              >
                <div className="flex items-center gap-2">Chapter {getSortIcon('chapter')}</div>
              </th>
              <th
                className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('baseline_time_ms')}
              >
                <div className="flex items-center justify-end gap-2">
                  Baseline {getSortIcon('baseline_time_ms')}
                </div>
              </th>
              <th
                className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('optimized_time_ms')}
              >
                <div className="flex items-center justify-end gap-2">
                  Optimized {getSortIcon('optimized_time_ms')}
                </div>
              </th>
              <th
                className="px-5 py-3 text-right text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('speedup')}
              >
                <div className="flex items-center justify-end gap-2">
                  Speedup {getSortIcon('speedup')}
                </div>
              </th>
              <th
                className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase cursor-pointer hover:text-white"
                onClick={() => handleSort('status')}
              >
                <div className="flex items-center justify-center gap-2">
                  Status {getSortIcon('status')}
                </div>
              </th>
              <th className="px-5 py-3 text-center text-xs font-medium text-white/50 uppercase">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {loading &&
              Array.from({ length: Math.min(6, pageSize) }).map((_, index) => (
                <tr key={`skeleton-${index}`} className="animate-pulse">
                  <td className="px-5 py-4" colSpan={columnCount}>
                    <div className="h-3 bg-white/10 rounded-sm w-full" />
                  </td>
                </tr>
              ))}
            {!loading && error && (
              <tr>
                <td className="px-5 py-6 text-center text-sm text-accent-danger" colSpan={columnCount}>
                  {error}
                </td>
              </tr>
            )}
            {!loading && !error && benchmarks.length === 0 && (
              <tr>
                <td className="px-5 py-6 text-center text-sm text-white/50" colSpan={columnCount}>
                  No benchmarks match the current filters.
                </td>
              </tr>
            )}
            {!loading &&
              !error &&
              benchmarks.map((b, i) => {
                const key = getBenchmarkKey(b);
                const isPinned = pinnedBenchmarks.has(key);
                const displaySpeedup = b.speedup || 0;
                const rawSpeedup = b.raw_speedup ?? displaySpeedup;
                const isCapped = b.speedup_capped && rawSpeedup !== displaySpeedup;
                const isHighlighted = highlightKey && (b.name === highlightKey || key === highlightKey);
                return (
                  <tr
                    key={`${b.chapter}-${b.name}-${i}`}
                    className={cn(
                      'hover:bg-white/2 transition-colors',
                      isPinned && 'bg-accent-primary/5 border-l-2 border-accent-primary',
                      isHighlighted && 'bg-accent-secondary/10'
                    )}
                  >
                    {(onTogglePin || onToggleFavorite) && (
                      <td className="px-3 py-4">
                        <div className="flex items-center gap-2">
                          {onTogglePin && (
                            <button
                              onClick={() => onTogglePin(key)}
                              className={cn(
                                'p-1 rounded-sm transition-all',
                                isPinned
                                  ? 'text-accent-primary'
                                  : 'text-white/20 hover:text-white/60'
                              )}
                              title={isPinned ? 'Unpin' : 'Pin'}
                            >
                              <Pin className="w-4 h-4" />
                            </button>
                          )}
                          {onToggleFavorite && (
                            <button
                              onClick={() => onToggleFavorite(key)}
                              className={cn(
                                'p-1 rounded-sm transition-all',
                                favorites.has(key)
                                  ? 'text-accent-secondary'
                                  : 'text-white/20 hover:text-white/60'
                              )}
                              title={favorites.has(key) ? 'Unfavorite' : 'Favorite'}
                            >
                              <Star className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    )}
                    <td className="px-5 py-4">
                      <span className="font-medium text-white">{b.name}</span>
                    </td>
                    <td className="px-5 py-4">
                      <span className="px-2 py-1 bg-accent-secondary/10 text-accent-secondary rounded-sm text-xs">
                        {b.chapter}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-right font-mono text-sm text-accent-tertiary">
                      {b.status === 'succeeded' ? formatMs(b.baseline_time_ms) : '-'}
                    </td>
                    <td className="px-5 py-4 text-right font-mono text-sm text-accent-success">
                      {b.status === 'succeeded' ? formatMs(b.optimized_time_ms) : '-'}
                    </td>
                    <td className="px-5 py-4 text-right">
                      {b.status === 'succeeded' && displaySpeedup ? (
                        <div className="flex items-center justify-end gap-2">
                          <span
                            className="font-bold text-lg"
                            style={{ color: getSpeedupColor(displaySpeedup) }}
                          >
                            {displaySpeedup.toFixed(2)}x
                          </span>
                          {isCapped && (
                            <span
                              className="text-[10px] text-white/50 bg-white/5 px-1.5 py-0.5 rounded-sm"
                              title={`Raw ${rawSpeedup.toFixed(2)}x${speedupCap ? ` • capped at ${speedupCap.toFixed(0)}x for display` : ''}`}
                            >
                              capped
                            </span>
                          )}
                        </div>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="px-5 py-4 text-center">
                      <span
                        className={cn(
                          'px-2 py-1 rounded-full text-xs font-medium',
                          b.status === 'succeeded' && 'bg-accent-success/20 text-accent-success',
                          b.status === 'failed' && 'bg-accent-danger/20 text-accent-danger',
                          b.status === 'skipped' && 'bg-white/10 text-white/50'
                        )}
                      >
                        {b.status}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-center">
                      <div className="flex items-center justify-center gap-2">
                        {onFocusBenchmark && (
                          <button
                            onClick={() => onFocusBenchmark(b)}
                            className="p-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
                            title="Focus mode"
                          >
                            <Target className="w-4 h-4" />
                          </button>
                        )}
                        {onShowCodeDiff && (
                          <button
                            onClick={() => onShowCodeDiff(b)}
                            className="p-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
                            title="View code diff"
                          >
                            <FileCode2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>

      <div className="px-5 py-4 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-white/50">
        <div>
          {total ? `Showing ${startIndex}-${endIndex} of ${total}` : 'No results'}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white/60 disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-white/60">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white/60 disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
