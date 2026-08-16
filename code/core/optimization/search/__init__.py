"""Optimization search module.

MCTS is an explicit heuristic simulator. It does not produce benchmark evidence.
Use the campaign executor and trusted benchmark harness for measured experiments.

Components:
- MCTSOptimizer: Monte Carlo Tree Search for compound optimization discovery
- LLMOracle: LLM-guided optimization suggestions with learned context
- UnifiedOptimizer: Combines MCTS + LLM + heuristics

Usage:
    from core.optimization.search import search_optimal_config

    result = search_optimal_config(
        model_config={"parameters_billions": 70, ...},
        hardware_config={"num_gpus": 8, "gpu_arch": "hopper", ...},
        optimization_goal="throughput",
        budget=100,
        simulation=True,
    )
"""

from .llm_oracle import (
    ContextCollector,
    LLMOracle,
    OptimizationSuggestion,
    OracleKnowledgeBase,
    ask_oracle,
    get_suggestions,
)
from .mcts_optimizer import (
    ActionLibrary,
    MCTSOptimizer,
    OptimizationAction,
    OptimizationDomain,
    OptimizationState,
    search_optimal_config,
)

__all__ = [
    # MCTS
    "MCTSOptimizer",
    "OptimizationAction",
    "OptimizationState",
    "OptimizationDomain",
    "ActionLibrary",
    "search_optimal_config",
    # LLM Oracle
    "LLMOracle",
    "OptimizationSuggestion",
    "OracleKnowledgeBase",
    "ContextCollector",
    "get_suggestions",
    "ask_oracle",
]
