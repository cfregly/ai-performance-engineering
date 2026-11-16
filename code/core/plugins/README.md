## aipe plugins

Core `aisp` CLI looks for Typer apps exposed via the `aipe.plugins` entry point group. This lets you add or remove entire command groups without modifying the core repository.

### How to publish a plugin (in another package)

In your extension package (e.g., `aipe_ext`):

```python
# aipe_ext/custom_plugin.py
import typer
from core.capabilities import register_capability

register_capability("cli.ext", provider="aipe_ext")
register_capability("custom", provider="aipe_ext")

app = typer.Typer(help="Custom analysis commands")

@app.command()
def inspect():
    typer.echo("Run custom analysis here")
```

And in `pyproject.toml` (or setup.cfg):

```toml
[project.entry-points."aipe.plugins"]
custom = "aipe_ext.custom_plugin:app"
```

When installed, `aisp` will auto-attach the plugin’s Typer app under the given name (`custom` in this example). Capabilities can be registered at import time to enable gated features such as `bench.llm`.

### Gated capabilities

- `cli.ext` — enables extension categories and commands such as ai/analyze/optimize/distributed/inference/training/monitor/report/profile/hf/cluster and mcp/tui.
- `bench.llm` — enables LLM analysis/auto-patching flags in `bench`.
- Additional capabilities are free-form (`custom`, `optimization_search`, etc.).

If a plugin fails to load, core CLI continues to function with only the core commands. Plugins should be defensive and avoid raising during import.
