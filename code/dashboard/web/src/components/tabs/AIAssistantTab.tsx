'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  Bot, 
  Cpu, 
  Server, 
  Zap, 
  Brain, 
  Activity, 
  TestTube,
  FileText,
  RefreshCw,
  Loader2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Play,
  Copy,
  Check,
  Search,
  Sparkles,
  HelpCircle,
} from 'lucide-react';
import { getAiTools, executeAiTool } from '@/lib/api';
import { cn, formatBytes, mebibytesToBytes } from '@/lib/utils';
import { useToast } from '@/components/Toast';
import type { GpuInfo } from '@/types';

interface McpTool {
  name: string;
  description: string;
  category: string;
  schema: {
    type: string;
    properties?: Record<string, { type: string; description?: string; default?: unknown }>;
    required?: string[];
  };
}

interface ToolResult {
  success: boolean;
  tool: string;
  result?: unknown;
  error?: string;
}

const categoryIcons: Record<string, React.ReactNode> = {
  gpu: <Cpu className="w-4 h-4" />,
  system: <Server className="w-4 h-4" />,
  analysis: <Activity className="w-4 h-4" />,
  optimization: <Zap className="w-4 h-4" />,
  distributed: <Server className="w-4 h-4" />,
  cluster: <Server className="w-4 h-4" />,
  inference: <Brain className="w-4 h-4" />,
  ai: <Bot className="w-4 h-4" />,
  profiling: <Activity className="w-4 h-4" />,
  benchmarks: <TestTube className="w-4 h-4" />,
  tests: <TestTube className="w-4 h-4" />,
  exports: <FileText className="w-4 h-4" />,
  other: <Sparkles className="w-4 h-4" />,
};

const categoryColors: Record<string, string> = {
  gpu: 'text-accent-success',
  system: 'text-accent-info',
  analysis: 'text-accent-warning',
  optimization: 'text-accent-primary',
  distributed: 'text-accent-danger',
  cluster: 'text-accent-info',
  inference: 'text-accent-secondary',
  ai: 'text-accent-tertiary',
  profiling: 'text-accent-warning',
  benchmarks: 'text-accent-success',
  tests: 'text-accent-info',
  exports: 'text-white/60',
  other: 'text-white/40',
};

const priorityStyles: Record<string, string> = {
  high: 'bg-accent-danger/20 text-accent-danger border-accent-danger/30',
  medium: 'bg-accent-warning/20 text-accent-warning border-accent-warning/30',
  low: 'bg-accent-success/20 text-accent-success border-accent-success/30',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function extractGpuInfo(payload: unknown): GpuInfo | null {
  if (!payload) return null;
  if (isRecord(payload) && isRecord(payload.gpu)) {
    return payload.gpu as unknown as GpuInfo;
  }
  if (isRecord(payload) && 'memory_total' in payload && 'utilization' in payload) {
    return payload as unknown as GpuInfo;
  }
  return null;
}

function renderGpuSummary(gpu: GpuInfo) {
  const memoryTotal = typeof gpu.memory_total === 'number' ? mebibytesToBytes(gpu.memory_total) : 0;
  const memoryUsed = typeof gpu.memory_used === 'number' ? mebibytesToBytes(gpu.memory_used) : 0;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-white/70">
      <div>
        <div className="text-white/40 mb-1">GPU</div>
        <div className="text-white">{gpu.name}</div>
      </div>
      <div>
        <div className="text-white/40 mb-1">Memory</div>
        <div className="text-white">
          {formatBytes(memoryUsed)} / {formatBytes(memoryTotal)}
        </div>
      </div>
      <div>
        <div className="text-white/40 mb-1">Utilization</div>
        <div className="text-white">{gpu.utilization}%</div>
      </div>
      <div>
        <div className="text-white/40 mb-1">Temperature</div>
        <div className="text-white">{gpu.temperature}°C</div>
      </div>
    </div>
  );
}

function renderRecommendations(recommendations: unknown) {
  if (!Array.isArray(recommendations) || recommendations.length === 0) return null;
  return (
    <div className="space-y-2">
      {recommendations.map((rec, index) => {
        if (typeof rec === 'string') {
          return (
            <div key={`rec-${index}`} className="flex items-start gap-2 text-sm text-white/80">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-accent-primary" />
              <span>{rec}</span>
            </div>
          );
        }
        if (isRecord(rec)) {
          const priority = String(rec.priority || rec.severity || '').toLowerCase();
          const badgeClass = priorityStyles[priority] || 'bg-white/10 text-white/60 border-white/10';
          const titleRaw = rec.title ?? rec.name ?? `Recommendation ${index + 1}`;
          const title = typeof titleRaw === 'string' ? titleRaw : String(titleRaw);
          const descriptionRaw = rec.description;
          const description = descriptionRaw === undefined || descriptionRaw === null ? '' : String(descriptionRaw);
          return (
            <div key={`rec-${index}`} className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-medium text-white">
                  {title}
                </div>
                {priority && (
                  <span className={`text-[10px] uppercase px-2 py-0.5 rounded-full border ${badgeClass}`}>
                    {priority}
                  </span>
                )}
              </div>
              {description && <div className="text-xs text-white/60 mt-1">{description}</div>}
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

function renderKeyValue(payload: Record<string, unknown>) {
  const entries = Object.entries(payload).slice(0, 8);
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-white/70">
      {entries.map(([key, value]) => (
        <div key={key} className="flex items-start justify-between gap-2">
          <span className="text-white/40">{key}</span>
          <span className="text-white break-all">{String(value)}</span>
        </div>
      ))}
    </div>
  );
}

export function AIAssistantTab() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tools, setTools] = useState<McpTool[]>([]);
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [mcpStatus, setMcpStatus] = useState<{ available: boolean; tools_count: number } | null>(null);
  const [selectedTool, setSelectedTool] = useState<McpTool | null>(null);
  const [toolParams, setToolParams] = useState<Record<string, unknown>>({});
  const [executing, setExecuting] = useState(false);
  const [results, setResults] = useState<ToolResult[]>([]);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['gpu', 'ai', 'system']));
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedResult, setCopiedResult] = useState<number | null>(null);
  const { showToast } = useToast();

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const toolsData = await getAiTools();
      const toolsResult = toolsData as any;
      setTools(toolsResult.tools || []);
      setCategories(toolsResult.categories || {});
      setMcpStatus({
        available: toolsResult.available !== false,
        tools_count: toolsResult.count || toolsResult.tools?.length || 0,
      });
      
      if (toolsResult.error) {
        setError(toolsResult.error);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load MCP tools');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const handleToolSelect = (tool: McpTool) => {
    setSelectedTool(tool);
    // Initialize params with defaults
    const defaults: Record<string, unknown> = {};
    if (tool.schema.properties) {
      for (const [key, prop] of Object.entries(tool.schema.properties)) {
        if (prop.default !== undefined) {
          defaults[key] = prop.default;
        }
      }
    }
    setToolParams(defaults);
  };

  const handleExecuteTool = async () => {
    if (!selectedTool) return;
    
    try {
      setExecuting(true);
      const result = await executeAiTool(selectedTool.name, toolParams) as ToolResult;
      setResults(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 results
      
      if (result.success) {
        showToast(`${selectedTool.name} completed`, 'success');
      } else {
        showToast(`${selectedTool.name} failed: ${result.error}`, 'error');
      }
    } catch (e) {
      const errorResult: ToolResult = {
        success: false,
        tool: selectedTool.name,
        error: e instanceof Error ? e.message : 'Unknown error',
      };
      setResults(prev => [errorResult, ...prev.slice(0, 9)]);
      showToast('Tool execution failed', 'error');
    } finally {
      setExecuting(false);
    }
  };

  const copyResult = (index: number, result: unknown) => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopiedResult(index);
    setTimeout(() => setCopiedResult(null), 2000);
    showToast('Copied to clipboard', 'success');
  };

  const renderResultContent = (result: ToolResult) => {
    const payload = result.success ? result.result : { error: result.error };
    const errorMessage =
      (isRecord(payload) && typeof payload.error === 'string' && payload.error) ||
      (typeof result.error === 'string' ? result.error : '');

    if (errorMessage) {
      return (
        <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-accent-danger">
          {errorMessage}
        </div>
      );
    }

    const gpuInfo = extractGpuInfo(payload);
    if (gpuInfo) {
      return (
        <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
          {renderGpuSummary(gpuInfo)}
        </div>
      );
    }

    if (isRecord(payload)) {
      const summaryText = typeof payload.summary === 'string' ? payload.summary : null;
      const recommendations =
        payload.recommendations || payload.priority_recommendations || payload.actions;
      const keyFindings = payload.key_findings || payload.findings;

      return (
        <div className="space-y-3">
          {summaryText && (
            <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/80">
              {summaryText}
            </div>
          )}
          {Array.isArray(keyFindings) && keyFindings.length > 0 && (
            <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
              <div className="text-xs uppercase text-white/40 mb-2">Key findings</div>
              {renderRecommendations(keyFindings)}
            </div>
          )}
          {renderRecommendations(recommendations)}
          {!summaryText && !recommendations && !keyFindings && renderKeyValue(payload)}
          <details className="text-xs text-white/40">
            <summary className="cursor-pointer hover:text-white/70">Raw JSON</summary>
            <pre className="mt-2 text-xs text-white/60 bg-black/20 rounded-lg p-3 overflow-x-auto max-h-48">
              {JSON.stringify(payload, null, 2)}
            </pre>
          </details>
        </div>
      );
    }

    if (Array.isArray(payload)) {
      return (
        <div className="space-y-2">
          {payload.slice(0, 8).map((item, index) => (
            <div key={`array-${index}`} className="text-sm text-white/70">
              {typeof item === 'string' ? item : JSON.stringify(item)}
            </div>
          ))}
          <details className="text-xs text-white/40">
            <summary className="cursor-pointer hover:text-white/70">Raw JSON</summary>
            <pre className="mt-2 text-xs text-white/60 bg-black/20 rounded-lg p-3 overflow-x-auto max-h-48">
              {JSON.stringify(payload, null, 2)}
            </pre>
          </details>
        </div>
      );
    }

    return (
      <pre className="text-xs text-white/70 bg-black/20 rounded-lg p-3 overflow-x-auto max-h-48">
        {JSON.stringify(payload, null, 2)}
      </pre>
    );
  };

  const filteredTools = searchQuery
    ? tools.filter(t => 
        t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : tools;

  const filteredCategories = Object.entries(categories).filter(([cat, toolNames]) => {
    if (!searchQuery) return true;
    return toolNames.some(name => {
      const tool = tools.find(t => t.name === name);
      return tool && (
        tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tool.description.toLowerCase().includes(searchQuery.toLowerCase())
      );
    });
  });

  if (loading) {
    return (
      <div className="card">
        <div className="card-body space-y-4 animate-pulse">
          <div className="h-4 w-40 bg-white/10 rounded" />
          <div className="h-10 bg-white/5 rounded" />
          <div className="h-10 bg-white/5 rounded" />
          <div className="h-10 bg-white/5 rounded" />
        </div>
      </div>
    );
  }

  if (error && tools.length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center py-16">
          <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
          <p className="text-white/70 mb-4">{error}</p>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30 mx-auto"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-accent-secondary" />
            <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
            <span className="text-sm text-white/50">MCP Tools</span>
          </div>
          <div className="flex items-center gap-3">
            {mcpStatus && (
              <div className={cn(
                'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm',
                mcpStatus.available
                  ? 'bg-accent-success/20 text-accent-success'
                  : 'bg-accent-danger/20 text-accent-danger'
              )}>
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  mcpStatus.available ? 'bg-accent-success' : 'bg-accent-danger'
                )} />
                {mcpStatus.available ? `${mcpStatus.tools_count} tools` : 'Offline'}
              </div>
            )}
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
          </div>
        </div>
        
        {/* Search */}
        <div className="px-5 py-3 border-t border-white/5">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              type="text"
              placeholder="Search tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Tools List */}
        <div className="lg:col-span-1 space-y-2">
          {filteredCategories.map(([category, toolNames]) => {
            const isExpanded = expandedCategories.has(category);
            const categoryTools = toolNames
              .map(name => tools.find(t => t.name === name))
              .filter(Boolean) as McpTool[];
            
            const filteredCategoryTools = searchQuery
              ? categoryTools.filter(t => 
                  t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                  t.description.toLowerCase().includes(searchQuery.toLowerCase())
                )
              : categoryTools;
            
            if (filteredCategoryTools.length === 0) return null;
            
            return (
              <div key={category} className="card">
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full card-header hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className={categoryColors[category]}>
                      {categoryIcons[category]}
                    </span>
                    <span className="font-medium text-white capitalize">{category}</span>
                    <span className="text-xs text-white/40">({filteredCategoryTools.length})</span>
                  </div>
                  {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-white/40" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-white/40" />
                  )}
                </button>
                
                {isExpanded && (
                  <div className="border-t border-white/5">
                    {filteredCategoryTools.map((tool) => (
                      <button
                        key={tool.name}
                        onClick={() => handleToolSelect(tool)}
                        className={cn(
                          'w-full px-4 py-3 text-left hover:bg-white/5 transition-colors border-b border-white/5 last:border-b-0',
                          selectedTool?.name === tool.name && 'bg-accent-primary/10'
                        )}
                      >
                        <div className="text-sm font-mono text-accent-primary truncate">
                          {tool.name}
                        </div>
                        <div className="text-xs text-white/50 truncate mt-0.5">
                          {tool.description.slice(0, 80)}...
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Tool Configuration & Execution */}
        <div className="lg:col-span-2 space-y-4">
          {selectedTool ? (
            <>
              {/* Tool Details */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <h3 className="font-mono text-accent-primary">{selectedTool.name}</h3>
                    <p className="text-sm text-white/60 mt-1">{selectedTool.description}</p>
                  </div>
                  <span className={cn('px-2 py-1 rounded text-xs capitalize', categoryColors[selectedTool.category])}>
                    {selectedTool.category}
                  </span>
                </div>
                
                {/* Parameters */}
                {selectedTool.schema.properties && Object.keys(selectedTool.schema.properties).length > 0 && (
                  <div className="card-body border-t border-white/5">
                    <h4 className="text-sm font-medium text-white mb-3">Parameters</h4>
                    <div className="space-y-3">
                      {Object.entries(selectedTool.schema.properties).map(([key, prop]) => (
                        <div key={key}>
                          <label className="flex items-center gap-2 text-sm text-white/70 mb-1">
                            {key}
                            {selectedTool.schema.required?.includes(key) && (
                              <span className="text-accent-danger">*</span>
                            )}
                            {prop.description && (
                              <span title={prop.description}>
                                <HelpCircle className="w-3 h-3 text-white/30" />
                              </span>
                            )}
                          </label>
                          {prop.type === 'boolean' ? (
                            <label className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={Boolean(toolParams[key])}
                                onChange={(e) => setToolParams({ ...toolParams, [key]: e.target.checked })}
                                className="w-4 h-4 accent-accent-primary"
                              />
                              <span className="text-sm text-white/60">Enabled</span>
                            </label>
                          ) : prop.type === 'number' || prop.type === 'integer' ? (
                            <input
                              type="number"
                              value={toolParams[key] as number || ''}
                              onChange={(e) => setToolParams({ ...toolParams, [key]: Number(e.target.value) })}
                              placeholder={prop.default?.toString() || ''}
                              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
                            />
                          ) : (
                            <input
                              type="text"
                              value={toolParams[key] as string || ''}
                              onChange={(e) => setToolParams({ ...toolParams, [key]: e.target.value })}
                              placeholder={prop.default?.toString() || prop.description || ''}
                              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Execute Button */}
                <div className="px-5 py-4 border-t border-white/5">
                  <button
                    onClick={handleExecuteTool}
                    disabled={executing}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-accent-primary to-accent-secondary text-black rounded-lg font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
                  >
                    {executing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Executing...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Execute Tool
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Results */}
              {results.length > 0 && (
                <div className="card">
                  <div className="card-header">
                    <h3 className="font-medium text-white">Results</h3>
                    <span className="text-sm text-white/50">{results.length} executions</span>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {results.map((result, index) => (
                      <div
                        key={index}
                        className={cn(
                          'px-5 py-4 border-t border-white/5',
                          index === 0 && 'bg-accent-primary/5'
                        )}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className={cn(
                              'w-2 h-2 rounded-full',
                              result.success ? 'bg-accent-success' : 'bg-accent-danger'
                            )} />
                            <span className="font-mono text-sm text-accent-primary">{result.tool}</span>
                          </div>
                          <button
                            onClick={() => copyResult(index, result.result || result.error)}
                            className="p-1.5 hover:bg-white/5 rounded text-white/40 hover:text-white"
                          >
                            {copiedResult === index ? (
                              <Check className="w-4 h-4 text-accent-success" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                        {renderResultContent(result)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card">
              <div className="card-body text-center py-16">
                <Bot className="w-16 h-16 mx-auto mb-4 text-white/20" />
                <h3 className="text-lg font-medium text-white mb-2">Select a Tool</h3>
                <p className="text-white/50 max-w-md mx-auto">
                  Choose an AI tool from the list to configure and execute. Tools are organized by category
                  and provide GPU analysis, performance recommendations, and more.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Quick Actions</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { name: 'gpu_info', label: 'GPU Info', icon: Cpu, color: 'accent-success' },
              { name: 'recommend', label: 'Get Recommendations', icon: Sparkles, color: 'accent-primary' },
              { name: 'analyze_bottlenecks', label: 'Find Bottlenecks', icon: Activity, color: 'accent-warning' },
              { name: 'system_context', label: 'System Context', icon: Server, color: 'accent-info' },
            ].map((action) => {
              const tool = tools.find(t => t.name === action.name);
              const Icon = action.icon;
              return (
                <button
                  key={action.name}
                  onClick={() => tool && handleToolSelect(tool)}
                  className={cn(
                    'p-4 rounded-lg border transition-all text-left',
                    tool
                      ? 'bg-white/5 border-white/10 hover:border-accent-primary/30'
                      : 'bg-white/[0.02] border-white/5 opacity-50 cursor-not-allowed'
                  )}
                >
                  <Icon className={cn('w-6 h-6 mb-2', `text-${action.color}`)} />
                  <div className="font-medium text-white">{action.label}</div>
                  <div className="text-xs text-white/50 mt-1">
                    {tool ? 'Click to execute' : 'Not available'}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
