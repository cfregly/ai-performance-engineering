# AI Performance Dashboard (Next.js)

A modern, high-performance dashboard for GPU benchmark visualization and optimization insights.

## Quick Start

### Prerequisites

- Node.js 24 or newer
- Python 3.10 or newer

### Installation

```bash
cd dashboard/web
npm ci
```

### Development

From the `code/` directory, start the API and Next.js UI together:

```bash
make dashboard
```

The API listens on [http://127.0.0.1:6970](http://127.0.0.1:6970). The UI listens on [http://127.0.0.1:3000](http://127.0.0.1:3000) and proxies `/api/*` to the API.

To run either process by itself:

```bash
make dashboard-api
make dashboard-web
```

### Production Build

```bash
npm run build
npm start
```

## Architecture

```
src/
├── app/                          # Next.js App Router routes
│   ├── layout.tsx               # Root layout, fonts, animated background
│   ├── page.tsx                 # Overview page
│   ├── compare/page.tsx         # Benchmark compare workflow
│   ├── profiler/page.tsx        # Profile pair browser + comparison UI
│   ├── memory/page.tsx          # Memory route (currently marked coming soon)
│   ├── multi-gpu/page.tsx       # Topology and multi-GPU views
│   ├── history/page.tsx         # Historical benchmark runs
│   ├── tier1/page.tsx           # Tier-1 benchmark surfaces
│   ├── system/page.tsx          # System inspection / health tools
│   ├── cluster/page.tsx         # Cluster eval + promotion tools
│   ├── contracts/page.tsx       # BenchmarkRun contract surface
│   └── globals.css              # Tailwind + dashboard design tokens
├── components/
│   ├── DashboardShell.tsx       # Shared shell + keyboard navigation
│   ├── Navigation.tsx           # Route tab bar
│   ├── StatsCard.tsx            # Stat display cards
│   ├── SpeedupChart.tsx         # Speedup visualization
│   ├── StatusChart.tsx          # Status breakdown visualization
│   ├── BenchmarkTable.tsx       # Sortable/filterable benchmark table
│   ├── GpuCard.tsx              # Live GPU telemetry card
│   ├── SoftwareStackWidget.tsx  # Backend software snapshot
│   ├── DependenciesWidget.tsx   # Dependency health widget
│   └── tabs/AIAssistantTab.tsx  # MCP-backed AI assistant panel
├── lib/
│   ├── api.ts                   # API client functions
│   ├── useGpuStream.ts          # GPU polling hook
│   └── utils.ts                 # UI helper functions
└── types/index.ts               # TypeScript types
```

## Features

- **Overview**: stats, status/speedup charts, live GPU card, software/dependency widgets, benchmark table, and the AI assistant panel.
- **Compare**: baseline-vs-candidate benchmark comparison workflow.
- **Profiler**: profile-pair browser with Nsight-driven comparison views.
- **Multi-GPU / History / Tier-1 / System / Cluster / Contracts**: dedicated routes for topology, run history, tier-1 coverage, system inspection, cluster operations, and BenchmarkRun contract surfacing.
- **Memory**: route exists but is still explicitly labeled "Coming soon" in the UI.

## API Proxy

The Next.js app proxies all `/api/*` requests to the Python backend via `next.config.js`.
By default that proxy targets `http://127.0.0.1:6970`, and you can override it with
`BACKEND_HOST` / `BACKEND_PORT` (or their `NEXT_PUBLIC_...` variants).

## Benchmark Environment Mode Parity

Dashboard benchmark actions are backed by MCP benchmark tools and use the same validity flags as CLI/MCP:
- `validity_profile` (benchmark validity profile): `strict` (default; fail-fast with full validity checks) or `portable` (compatibility mode for virtualized/limited hosts)
- `allow_portable_expectations_update`: optional override if a portable run should update expectation files

Example backend payload:

```json
{
  "targets": ["ch10:atomic_reduction"],
  "profile": "minimal",
  "validity_profile": "strict"
}
```

## Styling

Uses Tailwind CSS with custom design tokens matching the current dashboard shell:

- Dark-first palette with glass-morphism cards and animated gradient/grid backgrounds
- Gradient backgrounds with subtle animation
- Accent colors: Cyan (`#00f5d4`), Purple (`#9d4edd`), Pink (`#f72585`)
- Space Grotesk for UI text, JetBrains Mono for code
