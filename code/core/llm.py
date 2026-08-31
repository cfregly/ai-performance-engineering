"""
🧠 Unified LLM Client - THE SINGLE SOURCE OF TRUTH

All LLM calls in the codebase should go through this module.
DO NOT create new LLM clients elsewhere!

Usage:
    from core.llm import llm_call, get_llm_status
    
    # Simple call
    response = llm_call("Explain flash attention")
    
    # With system prompt
    response = llm_call(
        prompt="Why is my kernel slow?",
        system="You are a GPU performance expert."
    )
    
    # Check status
    status = get_llm_status()
    print(f"Provider: {status['provider']}, Model: {status['model']}")

Supported backends (auto-detected from env):
    - OpenAI (OPENAI_API_KEY)
    - Anthropic (ANTHROPIC_API_KEY)
    - Ollama (OLLAMA_HOST)
    - vLLM (VLLM_API_BASE)

Anthropic defaults to claude-sonnet-4-6, the documented replacement for the
Sonnet 4 model retired on June 15, 2026. ANTHROPIC_MODEL, then PERF_LLM_MODEL,
can explicitly override it. Lifecycle checked August 30, 2026:
https://platform.claude.com/docs/en/about-claude/model-deprecations

The default output budget is 4096 tokens. LLM_MAX_TOKENS takes precedence over
PERF_LLM_MAX_TOKENS, and llm_call(max_tokens=...) overrides the configured budget.
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from core.utils.dotenv import load_repo_dotenv, reset_repo_dotenv_cache

# Find repo root for .env loading
CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """LLM configuration - loaded once from environment."""
    provider: str  # openai, anthropic, ollama, vllm
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load config from environment with provider-specific defaults."""
        _load_env()
        # Keep environment-backed and explicitly constructed configs consistent.
        env_max = int(
            os.environ.get("LLM_MAX_TOKENS")
            or os.environ.get("PERF_LLM_MAX_TOKENS")
            or cls.max_tokens
        )
        if env_max <= 0:
            raise ValueError("LLM_MAX_TOKENS/PERF_LLM_MAX_TOKENS must be a positive integer")

        def _select(provider: str) -> 'LLMConfig':
            provider = provider.lower()
            if provider == "anthropic":
                return cls(
                    provider="anthropic",
                    model=os.environ.get("ANTHROPIC_MODEL") or os.environ.get("PERF_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL,
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    base_url=None,
                    temperature=float(os.environ.get("PERF_LLM_TEMPERATURE", "0.7")),
                    max_tokens=env_max,
                )
            if provider == "openai":
                return cls(
                    provider="openai",
                    model=os.environ.get("OPENAI_MODEL") or os.environ.get("PERF_LLM_MODEL") or "gpt-4o",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=None,
                    temperature=float(os.environ.get("PERF_LLM_TEMPERATURE", "0.7")),
                    max_tokens=env_max,
                )
            if provider == "vllm":
                return cls(
                    provider="vllm",
                    model=os.environ.get("PERF_LLM_MODEL") or "default",
                    api_key=None,
                    base_url=os.environ.get("VLLM_API_BASE"),
                    temperature=float(os.environ.get("PERF_LLM_TEMPERATURE", "0.7")),
                    max_tokens=env_max,
                )
            if provider == "ollama":
                return cls(
                    provider="ollama",
                    model=os.environ.get("PERF_LLM_MODEL") or "llama3.2",
                    api_key=None,
                    base_url=os.environ.get("OLLAMA_HOST"),
                    temperature=float(os.environ.get("PERF_LLM_TEMPERATURE", "0.7")),
                    max_tokens=env_max,
                )
            return cls(provider="none", model="none", api_key=None, base_url=None, temperature=0.7, max_tokens=env_max)

        provider = (os.environ.get("PERF_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "auto").lower()

        if provider not in ("", "auto", "none"):
            return _select(provider)

        if os.environ.get("ANTHROPIC_API_KEY"):
            return _select("anthropic")
        if os.environ.get("OPENAI_API_KEY"):
            return _select("openai")
        if os.environ.get("VLLM_API_BASE"):
            return _select("vllm")
        if os.environ.get("OLLAMA_HOST") or _check_ollama():
            return _select("ollama")

        return _select("none")


def _load_env():
    """Load .env files (idempotent).
    
    Loads .env first, then .env.local (which takes precedence).
    Always loads from .env files even if variables are already set in os.environ,
    to ensure .env file values are used.
    """
    _load_env._loaded = getattr(_load_env, '_loaded', False)
    if _load_env._loaded:
        return
    _load_env._loaded = True
    load_repo_dotenv(CODE_ROOT)


# Load .env files at module import time to ensure they're available
_load_env()


def _check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.error
        import urllib.request
        url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434') + '/api/tags'
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


# =============================================================================
# SINGLETON CONFIG
# =============================================================================

_config: Optional[LLMConfig] = None

def get_config() -> LLMConfig:
    """Get the singleton LLM config."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def reset_config():
    """Reset config (useful for testing or reloading after .env changes)."""
    global _config
    _config = None
    # Reset the _load_env flag so it reloads .env files
    _load_env._loaded = False
    reset_repo_dotenv_cache(CODE_ROOT)


# =============================================================================
# LLM CALL - THE SINGLE ENTRY POINT
# =============================================================================

def llm_call(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = True,
) -> str:
    """
    Make an LLM call. This is THE function all code should use.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        temperature: Override default temperature
        max_tokens: Positive integer overriding the configured output-token budget.
            Provider/model limits still apply; this client does not raise or clamp the budget.
        json_mode: Request JSON output (OpenAI/Anthropic)
    
    Returns:
        The LLM response text
    
    Raises:
        RuntimeError: If no LLM backend is available
        ValueError: If the output-token budget is not a positive integer
    """
    config = get_config()
    
    if config.provider == 'none':
        raise RuntimeError("No LLM backend configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or configure Ollama.")
    
    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    
    if config.provider == 'openai':
        return _call_openai(prompt, system, temp, tokens, json_mode, config)
    elif config.provider == 'anthropic':
        return _call_anthropic(prompt, system, temp, tokens, config)
    elif config.provider == 'ollama':
        return _call_ollama(prompt, system, temp, tokens, config)
    elif config.provider == 'vllm':
        return _call_vllm(prompt, system, temp, tokens, config)
    else:
        raise RuntimeError(f"Unknown LLM provider: {config.provider}")


def _call_openai(prompt: str, system: Optional[str], temperature: float,
                 max_tokens: int, json_mode: bool, config: LLMConfig) -> str:
    """Call OpenAI API with compatibility for new Responses API (gpt-5.x)."""
    import json as _json
    import urllib.request
    import openai

    model_lower = config.model.lower()
    use_responses_api = model_lower.startswith(("gpt-5", "gpt-4.1", "o1", "o3", "o4"))

    if use_responses_api:
        # Build Responses API payload
        input_msgs = []
        if system:
            input_msgs.append({
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            })
        input_msgs.append({
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        })

        body = {
            "model": config.model,
            "input": input_msgs,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=_json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = _json.loads(resp.read().decode())
                output = parsed.get("output") or []
                texts = []
                for entry in output:
                    for block in entry.get("content", []):
                        if block.get("type") in {"output_text", "text", "input_text"}:
                            texts.append(block.get("text", ""))
                return "\n".join(t for t in texts if t)
        except Exception as e:
            detail = ""
            try:
                import urllib.error
                if isinstance(e, urllib.error.HTTPError) and e.read:
                    detail = e.read().decode()
            except (AttributeError, OSError, UnicodeDecodeError):
                pass
            raise RuntimeError(f"OpenAI API error: {e} {detail}") from e

    # Legacy Chat Completions path
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = openai.OpenAI(api_key=config.api_key)
    kwargs = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e


def _call_anthropic(prompt: str, system: Optional[str], temperature: float,
                    max_tokens: int, config: LLMConfig) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.api_key)
        
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system:
            kwargs["system"] = system
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
        
    except Exception as e:
        raise RuntimeError(f"Anthropic API error: {e}")


def _call_ollama(prompt: str, system: Optional[str], temperature: float,
                 max_tokens: int, config: LLMConfig) -> str:
    """Call Ollama API."""
    import urllib.request
    
    base_url = config.base_url or 'http://localhost:11434'
    url = f"{base_url}/api/generate"
    
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    data = json.dumps({
        "model": config.model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get('response', '')
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def _call_vllm(prompt: str, system: Optional[str], temperature: float,
               max_tokens: int, config: LLMConfig) -> str:
    """Call vLLM OpenAI-compatible API."""
    import urllib.request
    
    base_url = config.base_url or 'http://localhost:8000'
    url = f"{base_url}/v1/chat/completions"
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    data = json.dumps({
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result['choices'][0]['message']['content']
    except Exception as e:
        raise RuntimeError(f"vLLM API error: {e}")


# =============================================================================
# STATUS & UTILITIES
# =============================================================================

def get_llm_status(probe: bool = False) -> Dict[str, Any]:
    """Get LLM backend status.
    
    Args:
        probe: When True, issue a real LLM call to verify connectivity.
    """
    # Ensure .env is loaded before checking config
    _load_env()
    
    config = get_config()
    available = config.provider not in ("none", "") and (
        bool(config.api_key) or bool(config.base_url)
    )
    probe_error = None

    if probe and available:
        try:
            llm_call("LLM connectivity check", max_tokens=64)
            available = True
        except Exception as exc:
            available = False
            probe_error = str(exc)

    status = {
        "available": available,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
    }
    if probe_error:
        status["error"] = probe_error
    status["probed"] = probe
    
    # Add warning and setup instructions when LLM is unavailable
    if not available:
        env_file = CODE_ROOT / ".env"
        env_local_file = CODE_ROOT / ".env.local"
        has_env = env_file.exists() or env_local_file.exists()
        
        setup_instructions = []
        if has_env:
            setup_instructions.append(
                f"⚠️  LLM backend unavailable despite .env file(s) present. "
                f"Please verify your API key is correctly set in {env_file.name} or {env_local_file.name}"
            )
        else:
            setup_instructions.append(
                f"⚠️  No LLM backend configured. Create a .env file in {CODE_ROOT} with one of:"
            )
        
        setup_instructions.extend([
            "",
            "To enable LLM analysis and patching, add ONE of the following to your .env file:",
            "",
            "  # Option 1: OpenAI",
            "  OPENAI_API_KEY=your-api-key-here",
            "",
            "  # Option 2: Anthropic",
            "  ANTHROPIC_API_KEY=your-api-key-here",
            "",
            "  # Option 3: Local Ollama (if running)",
            "  OLLAMA_HOST=http://localhost:11434",
            "",
            "  # Option 4: vLLM server",
            "  VLLM_API_BASE=http://localhost:8000/v1",
            "",
            "Optional: Set provider explicitly",
            "  PERF_LLM_PROVIDER=openai  # or 'anthropic', 'ollama', 'vllm'",
            "",
            f"After adding your API key, restart the MCP server or call reset_config() to reload.",
        ])
        
        status["warning"] = "\n".join(setup_instructions)
        status["setup_required"] = True
    
    return status


def is_available() -> bool:
    """Quick check if LLM is available."""
    config = get_config()
    return config.provider != 'none'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

PERF_EXPERT_SYSTEM = """You are an expert GPU performance engineer specializing in:
- CUDA kernel optimization
- PyTorch and deep learning frameworks
- Distributed training (FSDP, tensor/pipeline parallelism)
- LLM inference optimization (vLLM, TensorRT-LLM)
- Memory optimization (gradient checkpointing, mixed precision)

Provide concise, actionable advice with code examples when helpful.
Reference specific techniques from the AI Performance Engineering book when relevant."""


def ask_performance_question(question: str, context: Optional[str] = None) -> str:
    """Ask a GPU performance question with expert system prompt."""
    prompt = question
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    return llm_call(prompt, system=PERF_EXPERT_SYSTEM)


def explain_concept(concept: str) -> str:
    """Explain a GPU/AI performance concept."""
    system = PERF_EXPERT_SYSTEM + "\n\nExplain concepts clearly with: what it is, when to use it, key parameters, common pitfalls, and a code example."
    return llm_call(f"Explain: {concept}", system=system)


def analyze_code(code: str, goal: str = "optimize") -> str:
    """Analyze code for performance improvements."""
    prompt = f"""Analyze this code and suggest {goal} improvements:

```python
{code}
```

Provide specific, actionable recommendations with code examples."""
    
    return llm_call(prompt, system=PERF_EXPERT_SYSTEM)
