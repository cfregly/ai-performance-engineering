"""Plugin loader for aisp extensions.

Plugins can expose Typer sub-apps and register capabilities via entry points:

    entry_points={
        "aipe.plugins": [
            "custom_tools = aipe_ext.tools:typer_app",
        ],
    }

The entry point object should be either:
  - a Typer app instance, or
  - a callable returning a Typer app, or
  - a tuple of (name, app, description)
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, List, Optional

from core.capabilities import register_capability


PLUGIN_GROUP = "aipe.plugins"


@dataclass
class PluginApp:
    name: str
    app: Any
    description: str = ""


def _coerce_plugin_app(obj: Any, ep_name: str) -> Optional[PluginApp]:
    """Normalize plugin entry point objects into PluginApp instances."""
    try:
        if callable(obj) and not hasattr(obj, "info"):  # likely a factory
            obj = obj()
    except Exception:
        return None

    if isinstance(obj, PluginApp):
        return obj

    # Tuple form: (name, app, description?)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        name = str(obj[0])
        app = obj[1]
        desc = str(obj[2]) if len(obj) >= 3 else ""
        return PluginApp(name=name, app=app, description=desc)

    # Direct Typer app; use entry point name as the command name
    if obj is not None:
        return PluginApp(name=ep_name.replace("_", "-"), app=obj, description="")

    return None


def load_plugin_apps() -> List[PluginApp]:
    """Discover and load Typer plugin apps via entry points."""
    apps: List[PluginApp] = []
    try:
        eps = importlib.metadata.entry_points()
        candidates = eps.select(group=PLUGIN_GROUP) if hasattr(eps, "select") else eps.get(PLUGIN_GROUP, [])
        for ep in candidates:
            try:
                obj = ep.load()
                plugin = _coerce_plugin_app(obj, ep.name)
                if plugin and plugin.app:
                    apps.append(plugin)
                    # Optional: allow plugin to self-register capability
                    register_capability(f"plugin:{ep.name}", provider=ep.value)
            except Exception:
                # Swallow plugin import errors to avoid breaking core CLI
                continue
    except Exception:
        pass
    return apps
