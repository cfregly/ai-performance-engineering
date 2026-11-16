"""LLM-backed performance analysis engine used by CLI and MCP tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from core.llm import llm_call, LLMConfig, get_config


class PerformanceAnalysisEngine:
    """Thin wrapper that feeds structured prompts through the unified LLM client."""

    def __init__(self) -> None:
        # Force-load config to surface missing env early
        self.config: LLMConfig = get_config()

    def ask(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        payload = {"prompt": prompt, "context": context or {}}
        return llm_call(json.dumps(payload, default=str))

    def analyze_profile(
        self,
        profile_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        workload_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = "Analyze GPU profiling data and propose optimizations."
        context = {
            "profile_data": profile_data,
            "constraints": constraints or {},
            "workload": workload_info or {},
        }
        return self.ask(prompt, context)

    def analyze_distributed(
        self,
        cluster_info: Dict[str, Any],
        performance_data: Dict[str, Any],
        training_config: Dict[str, Any],
        comm_patterns: Dict[str, Any],
    ) -> str:
        prompt = "Analyze distributed training performance and communication."
        context = {
            "cluster": cluster_info,
            "performance": performance_data,
            "training_config": training_config,
            "communication": comm_patterns,
        }
        return self.ask(prompt, context)

    def analyze_inference(
        self,
        model_info: Dict[str, Any],
        serving_config: Dict[str, Any],
        metrics: Dict[str, Any],
        traffic_pattern: Dict[str, Any],
    ) -> str:
        prompt = "Analyze inference serving performance and optimize for latency/throughput."
        context = {
            "model": model_info,
            "serving": serving_config,
            "metrics": metrics,
            "traffic": traffic_pattern,
        }
        return self.ask(prompt, context)

    def analyze_rlhf(
        self,
        model_config: Dict[str, Any],
        algorithm: str,
        actor_info: Dict[str, Any],
        critic_info: Dict[str, Any],
        reference_info: Dict[str, Any],
        reward_info: Dict[str, Any],
        performance_data: Dict[str, Any],
        memory_usage: Dict[str, Any],
    ) -> str:
        prompt = "Analyze RLHF training setup and propose efficiency improvements."
        context = {
            "model_config": model_config,
            "algorithm": algorithm,
            "actor": actor_info,
            "critic": critic_info,
            "reference": reference_info,
            "reward": reward_info,
            "performance": performance_data,
            "memory": memory_usage,
        }
        return self.ask(prompt, context)

