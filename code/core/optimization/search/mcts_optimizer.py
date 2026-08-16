#!/usr/bin/env python3
"""Monte Carlo Tree Search for simulated optimization exploration.

This module ranks compound optimization ideas with a heuristic simulator. It is
not a benchmark runner and its output is not performance evidence. Callers must
opt in with ``simulation=True``. Measured promotion belongs in the campaign
executor and trusted benchmark harness.

The simulator explores:
Uses MCTS with UCB exploration to find the best combination of:
- Parallelism configurations (TP, PP, DP, CP, EP)
- Precision modes (FP32, BF16, FP8)
- Checkpointing strategies
- Communication optimizations
- Kernel fusion patterns
- Memory-efficient optimizers

The search engine:
1. Models the optimization space as a tree
2. Uses UCB1 to balance exploration vs exploitation
3. Scores configurations with documented heuristic estimates
4. Can incorporate learned simulation priors
5. Persists context-scoped simulation knowledge for future sessions

Usage:
    from core.optimization.search import MCTSOptimizer

    optimizer = MCTSOptimizer(hardware_config, model_config, simulation=True)
    result = optimizer.search(
        budget=100,  # Number of rollouts
        optimization_goal="throughput"
    )
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

KNOWLEDGE_SCHEMA_VERSION = 2


# =============================================================================
# OPTIMIZATION ACTIONS (Moves in the search space)
# =============================================================================


class OptimizationDomain(Enum):
    """Domains of optimization."""

    PARALLELISM = "parallelism"
    PRECISION = "precision"
    CHECKPOINTING = "checkpointing"
    OPTIMIZER = "optimizer"
    COMMUNICATION = "communication"
    KERNELS = "kernels"
    MEMORY = "memory"
    SCHEDULING = "scheduling"


@dataclass
class OptimizationAction:
    """A single optimization action that can be applied."""

    name: str
    domain: OptimizationDomain
    params: dict[str, Any]
    prerequisites: list[str] = field(default_factory=list)  # Required other actions
    conflicts: list[str] = field(default_factory=list)  # Incompatible actions
    estimated_memory_delta_gb: float = 0.0  # Memory impact
    estimated_throughput_delta_pct: float = 0.0  # Throughput impact
    hardware_requirements: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptimizationAction):
            return NotImplemented
        return self.name == other.name


# =============================================================================
# OPTIMIZATION STATE (Node in the search tree)
# =============================================================================


@dataclass
class OptimizationState:
    """
    State in the optimization search tree.
    Represents a configuration with applied optimizations.
    """

    applied_actions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # Estimated metrics (before actual evaluation)
    estimated_memory_gb: float = 0.0
    estimated_throughput_tps: float = 0.0

    # Actual metrics (after evaluation)
    actual_memory_gb: float | None = None
    actual_throughput_tps: float | None = None
    actual_speedup: float | None = None

    # Validity
    is_valid: bool = True
    validity_reason: str = ""

    def get_hash(self, context: dict[str, Any] | None = None) -> str:
        """Return a stable hash for the state and optional evaluation context."""
        payload = {
            "actions": sorted(self.applied_actions),
            "config": self.config,
            "context": context or {},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def clone(self) -> OptimizationState:
        """Create a deep copy."""
        return OptimizationState(
            applied_actions=list(self.applied_actions),
            config=copy.deepcopy(self.config),
            estimated_memory_gb=self.estimated_memory_gb,
            estimated_throughput_tps=self.estimated_throughput_tps,
        )


# =============================================================================
# MCTS NODE
# =============================================================================


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""

    state: OptimizationState
    parent: MCTSNode | None = None
    children: dict[str, MCTSNode] = field(default_factory=dict)  # action_name -> child

    # Statistics
    visits: int = 0
    total_value: float = 0.0

    # Unexplored actions from this node
    untried_actions: list[str] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for selection."""
        if self.visits == 0:
            return float("inf")

        exploitation = self.avg_value
        if self.parent is None:
            return exploitation

        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


# =============================================================================
# ACTION LIBRARY (All possible optimization actions)
# =============================================================================


class ActionLibrary:
    """
    Library of all available optimization actions.
    Dynamically generates actions based on hardware capabilities.
    """

    def __init__(self, hardware_config: dict[str, Any], model_config: dict[str, Any]):
        self.hardware = hardware_config
        self.model = model_config
        self.actions: dict[str, OptimizationAction] = {}
        self._build_action_library()

    def _build_action_library(self):
        """Build the complete action library."""
        self._add_parallelism_actions()
        self._add_precision_actions()
        self._add_checkpointing_actions()
        self._add_optimizer_actions()
        self._add_communication_actions()
        self._add_kernel_actions()
        self._add_memory_actions()
        self._add_scheduling_actions()

    def _add_parallelism_actions(self):
        """Add parallelism configuration actions."""
        num_gpus = self.hardware.get("num_gpus", 8)
        has_nvlink = self.hardware.get("has_nvlink", True)

        # Tensor Parallelism
        for tp in [1, 2, 4, 8]:
            if tp <= num_gpus:
                throughput_delta = -5 * (tp - 1) if has_nvlink else -15 * (tp - 1)
                self.actions[f"tp_{tp}"] = OptimizationAction(
                    name=f"tp_{tp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"tensor_parallel": tp},
                    estimated_memory_delta_gb=-self.model.get("parameters_billions", 7)
                    * 2
                    * (1 - 1 / tp),
                    estimated_throughput_delta_pct=throughput_delta,
                    conflicts=[f"tp_{x}" for x in [1, 2, 4, 8] if x != tp],
                )

        # Pipeline Parallelism
        for pp in [1, 2, 4, 8]:
            if pp <= num_gpus:
                bubble_overhead = (pp - 1) / (pp * 4) * 100
                self.actions[f"pp_{pp}"] = OptimizationAction(
                    name=f"pp_{pp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"pipeline_parallel": pp},
                    estimated_memory_delta_gb=-self.model.get("parameters_billions", 7)
                    * 2
                    * (1 - 1 / pp),
                    estimated_throughput_delta_pct=-bubble_overhead if pp > 1 else 0,
                    conflicts=[f"pp_{x}" for x in [1, 2, 4, 8] if x != pp],
                )

        # Data Parallelism
        for dp in [1, 2, 4, 8, 16, 32, 64]:
            if dp <= num_gpus:
                self.actions[f"dp_{dp}"] = OptimizationAction(
                    name=f"dp_{dp}",
                    domain=OptimizationDomain.PARALLELISM,
                    params={"data_parallel": dp},
                    estimated_memory_delta_gb=0,  # DP doesn't reduce model memory
                    estimated_throughput_delta_pct=(dp - 1) * 90 / dp,  # Near-linear scaling
                    conflicts=[f"dp_{x}" for x in [1, 2, 4, 8, 16, 32, 64] if x != dp],
                )

    def _add_precision_actions(self):
        """Add precision mode actions."""
        gpu_arch = self.hardware.get("gpu_arch", "ampere").lower()
        supports_fp8 = gpu_arch in ["hopper", "blackwell", "h100", "h200", "b100", "b200", "gb200"]
        supports_bf16 = gpu_arch not in ["volta", "turing", "pascal"]

        self.actions["precision_fp32"] = OptimizationAction(
            name="precision_fp32",
            domain=OptimizationDomain.PRECISION,
            params={"precision": "fp32"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=0,
            conflicts=["precision_bf16", "precision_fp16", "precision_fp8"],
        )

        if supports_bf16:
            self.actions["precision_bf16"] = OptimizationAction(
                name="precision_bf16",
                domain=OptimizationDomain.PRECISION,
                params={"precision": "bf16"},
                estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
                estimated_throughput_delta_pct=40,
                conflicts=["precision_fp32", "precision_fp16", "precision_fp8"],
            )

        self.actions["precision_fp16"] = OptimizationAction(
            name="precision_fp16",
            domain=OptimizationDomain.PRECISION,
            params={"precision": "fp16"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=35,
            conflicts=["precision_fp32", "precision_bf16", "precision_fp8"],
        )

        if supports_fp8:
            self.actions["precision_fp8"] = OptimizationAction(
                name="precision_fp8",
                domain=OptimizationDomain.PRECISION,
                params={"precision": "fp8"},
                estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 3,
                estimated_throughput_delta_pct=80,
                conflicts=["precision_fp32", "precision_bf16", "precision_fp16"],
                hardware_requirements=["Hopper/Blackwell GPU"],
            )

    def _add_checkpointing_actions(self):
        """Add activation checkpointing actions."""
        self.actions["checkpoint_none"] = OptimizationAction(
            name="checkpoint_none",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": False},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=0,
            conflicts=["checkpoint_full", "checkpoint_selective", "checkpoint_block"],
        )

        self.actions["checkpoint_full"] = OptimizationAction(
            name="checkpoint_full",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": True, "policy": "full"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7)
            * 4,  # Significant savings
            estimated_throughput_delta_pct=-33,  # Recompute overhead
            conflicts=["checkpoint_none", "checkpoint_selective", "checkpoint_block"],
        )

        self.actions["checkpoint_selective"] = OptimizationAction(
            name="checkpoint_selective",
            domain=OptimizationDomain.CHECKPOINTING,
            params={"gradient_checkpointing": True, "policy": "selective"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=-15,
            conflicts=["checkpoint_none", "checkpoint_full", "checkpoint_block"],
        )

    def _add_optimizer_actions(self):
        """Add optimizer actions."""
        self.actions["optimizer_adamw"] = OptimizationAction(
            name="optimizer_adamw",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adamw"},
            estimated_memory_delta_gb=0,  # Baseline
            estimated_throughput_delta_pct=0,
            conflicts=["optimizer_adamw_8bit", "optimizer_adafactor", "optimizer_lion"],
        )

        self.actions["optimizer_adamw_8bit"] = OptimizationAction(
            name="optimizer_adamw_8bit",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adamw_8bit"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 6,
            estimated_throughput_delta_pct=-2,
            conflicts=["optimizer_adamw", "optimizer_adafactor", "optimizer_lion"],
        )

        self.actions["optimizer_adafactor"] = OptimizationAction(
            name="optimizer_adafactor",
            domain=OptimizationDomain.OPTIMIZER,
            params={"optimizer": "adafactor"},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 7.5,
            estimated_throughput_delta_pct=-5,
            conflicts=["optimizer_adamw", "optimizer_adamw_8bit", "optimizer_lion"],
        )

    def _add_communication_actions(self):
        """Add communication optimization actions."""
        self.actions["comm_overlap"] = OptimizationAction(
            name="comm_overlap",
            domain=OptimizationDomain.COMMUNICATION,
            params={"overlap_communication": True},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=10,
            prerequisites=["dp_2", "dp_4", "dp_8", "dp_16", "dp_32", "dp_64"],
        )

        self.actions["gradient_compression"] = OptimizationAction(
            name="gradient_compression",
            domain=OptimizationDomain.COMMUNICATION,
            params={"gradient_compression": "powersgd"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=5,
            prerequisites=["dp_2", "dp_4", "dp_8", "dp_16", "dp_32", "dp_64"],
        )

    def _add_kernel_actions(self):
        """Add kernel fusion actions."""
        gpu_arch = self.hardware.get("gpu_arch", "ampere").lower()
        supports_flash_attn = gpu_arch in [
            "ampere",
            "hopper",
            "blackwell",
            "a100",
            "h100",
            "h200",
            "b100",
            "b200",
            "gb200",
        ]

        if supports_flash_attn:
            self.actions["flash_attention"] = OptimizationAction(
                name="flash_attention",
                domain=OptimizationDomain.KERNELS,
                params={"flash_attention": True},
                estimated_memory_delta_gb=-2,
                estimated_throughput_delta_pct=20,
            )

        self.actions["torch_compile"] = OptimizationAction(
            name="torch_compile",
            domain=OptimizationDomain.KERNELS,
            params={"torch_compile": True, "mode": "reduce-overhead"},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=15,
        )

        self.actions["fused_kernels"] = OptimizationAction(
            name="fused_kernels",
            domain=OptimizationDomain.KERNELS,
            params={"fused_layer_norm": True, "fused_bias_gelu": True},
            estimated_memory_delta_gb=0,
            estimated_throughput_delta_pct=10,
        )

    def _add_memory_actions(self):
        """Add memory optimization actions."""
        self.actions["cpu_offload_optimizer"] = OptimizationAction(
            name="cpu_offload_optimizer",
            domain=OptimizationDomain.MEMORY,
            params={"offload_optimizer": True},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 8,
            estimated_throughput_delta_pct=-20,
        )

        self.actions["cpu_offload_params"] = OptimizationAction(
            name="cpu_offload_params",
            domain=OptimizationDomain.MEMORY,
            params={"offload_params": True},
            estimated_memory_delta_gb=-self.model.get("parameters_billions", 7) * 2,
            estimated_throughput_delta_pct=-40,
        )

    def _add_scheduling_actions(self):
        """Add pipeline scheduling actions."""
        self.actions["schedule_1f1b"] = OptimizationAction(
            name="schedule_1f1b",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "1f1b"},
            estimated_throughput_delta_pct=0,
            prerequisites=["pp_2", "pp_4", "pp_8"],
            conflicts=["schedule_interleaved", "schedule_zero_bubble"],
        )

        self.actions["schedule_interleaved"] = OptimizationAction(
            name="schedule_interleaved",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "interleaved", "virtual_stages": 2},
            estimated_throughput_delta_pct=10,
            prerequisites=["pp_2", "pp_4", "pp_8"],
            conflicts=["schedule_1f1b", "schedule_zero_bubble"],
        )

        self.actions["schedule_zero_bubble"] = OptimizationAction(
            name="schedule_zero_bubble",
            domain=OptimizationDomain.SCHEDULING,
            params={"pipeline_schedule": "zero_bubble"},
            estimated_memory_delta_gb=self.model.get("parameters_billions", 7) * 0.5,
            estimated_throughput_delta_pct=20,
            prerequisites=["pp_4", "pp_8"],
            conflicts=["schedule_1f1b", "schedule_interleaved"],
        )

    def get_valid_actions(self, state: OptimizationState) -> list[str]:
        """Get all valid actions from current state."""
        valid = []

        for name, action in self.actions.items():
            if name in state.applied_actions:
                continue

            # Check conflicts
            has_conflict = any(c in state.applied_actions for c in action.conflicts)
            if has_conflict:
                continue

            # Check prerequisites (at least one must be satisfied)
            if action.prerequisites:
                has_prereq = any(p in state.applied_actions for p in action.prerequisites)
                if not has_prereq:
                    continue

            # Check hardware requirements
            hardware_ok = True
            for req in action.hardware_requirements:
                if "hopper" in req.lower() or "blackwell" in req.lower():
                    gpu_arch = self.hardware.get("gpu_arch", "").lower()
                    if gpu_arch not in [
                        "hopper",
                        "blackwell",
                        "h100",
                        "h200",
                        "b100",
                        "b200",
                        "gb200",
                    ]:
                        hardware_ok = False
                        break

            if hardware_ok:
                valid.append(name)

        return valid


# =============================================================================
# MCTS OPTIMIZER
# =============================================================================


class MCTSOptimizer:
    """Monte Carlo Tree Search over explicitly simulated configurations."""

    def __init__(
        self,
        hardware_config: dict[str, Any],
        model_config: dict[str, Any],
        evaluator: Callable[[OptimizationState], float] | None = None,
        simulation: bool = False,
        evaluation_context: dict[str, Any] | None = None,
        exploration_constant: float = 1.414,
        knowledge_base_path: Path | None = None,
    ):
        if evaluator is not None:
            raise ValueError(
                "MCTS external evaluators are not evidence-backed. Use the campaign "
                "executor for measured experiments."
            )
        if not simulation:
            raise ValueError(
                "MCTS is simulation-only. Pass simulation=True for heuristic search, "
                "or use the campaign executor for measured experiments."
            )

        self.hardware = hardware_config
        self.model = model_config
        self.action_library = ActionLibrary(hardware_config, model_config)
        self.exploration_constant = exploration_constant
        self.evaluation_mode = "simulation"
        self.evaluation_context = dict(evaluation_context or {})

        # Knowledge base for persistent learning
        self.knowledge_base_path = (
            knowledge_base_path or Path.home() / ".cache" / "mcts_optimizer" / "knowledge.json"
        )
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_base = self._load_knowledge_base()

        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0

    def search(
        self,
        budget: int = 100,
        optimization_goal: str = "throughput",  # "throughput", "memory", "balanced"
        max_depth: int = 10,
        early_stop_threshold: float = 0.95,  # Stop if we find something this good
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run heuristic MCTS search over compound optimization ideas.

        Args:
            budget: Number of rollouts/simulations
            optimization_goal: What to optimize for
            max_depth: Maximum depth of search
            early_stop_threshold: Normalized score to trigger early stop
            verbose: Print progress

        Returns:
            Highest-scoring simulated configuration with statistics and an
            explicit marker that the result is not performance evidence.
        """
        if budget < 1:
            raise ValueError("budget must be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if optimization_goal not in {"throughput", "memory", "balanced"}:
            raise ValueError("optimization_goal must be throughput, memory, or balanced")

        start_time = time.time()

        # Initialize root
        initial_state = self._create_initial_state()
        root = MCTSNode(
            state=initial_state,
            untried_actions=self.action_library.get_valid_actions(initial_state),
        )

        best_score = float("-inf")
        best_node = root
        scores_history = []

        for i in range(budget):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_terminal and node.untried_actions and self._node_depth(node) < max_depth:
                node = self._expand(node)

            # 3. Simulation/Evaluation
            score = self._evaluate(node.state, optimization_goal)
            scores_history.append(score)

            # 4. Backpropagation
            self._backpropagate(node, score)

            # Track best
            if score > best_score:
                best_score = score
                best_node = node
                if verbose:
                    print(f"  Iteration {i + 1}/{budget}: New best score {score:.4f}")
                    print(f"    Actions: {node.state.applied_actions}")

            # Early stopping
            if score >= early_stop_threshold:
                if verbose:
                    print(f"  Early stop at iteration {i + 1} - found excellent configuration")
                break

        # Save learned knowledge
        self._save_knowledge_base()

        search_time = time.time() - start_time

        initial_memory_gb = initial_state.estimated_memory_gb
        return {
            "evaluation_mode": self.evaluation_mode,
            "performance_claim_allowed": False,
            "evidence_warning": (
                "Heuristic simulation only. Validate every candidate with the "
                "campaign executor and trusted benchmark harness."
            ),
            "best_config": best_node.state.config,
            "best_actions": best_node.state.applied_actions,
            "best_score": best_score,
            "simulated_throughput_delta_pct": best_node.state.estimated_throughput_tps,
            "simulated_memory_delta_gb": (best_node.state.estimated_memory_gb - initial_memory_gb),
            "search_statistics": {
                "total_iterations": len(scores_history),
                "total_evaluations": self.total_evaluations,
                "cache_hits": self.cache_hits,
                "search_time_seconds": search_time,
                "scores_history": scores_history[-20:],  # Last 20
            },
            "tree_statistics": {
                "root_visits": root.visits,
                "num_children": len(root.children),
                "avg_depth": self._get_avg_depth(root),
            },
            "recommendations": self._generate_recommendations(best_node),
        }

    def _create_initial_state(self) -> OptimizationState:
        """Create the initial state (no optimizations applied)."""
        model_params_b = self.model.get("parameters_billions", 7)
        return OptimizationState(
            applied_actions=[],
            config={},
            estimated_memory_gb=model_params_b * 20,  # Rough estimate: 20x params
            estimated_throughput_tps=0.0,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 until we reach unexpanded node."""
        while node.children and node.is_fully_expanded:
            # Select child with highest UCB1 score
            node = max(
                node.children.values(), key=lambda n: n.ucb1_score(self.exploration_constant)
            )
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        if not node.untried_actions:
            return node

        # Pick action (can use LLM prior here for smarter selection)
        action_name = self._select_action_to_try(node)
        node.untried_actions.remove(action_name)

        # Create child state
        action = self.action_library.actions[action_name]
        child_state = node.state.clone()
        child_state.applied_actions.append(action_name)
        child_state.config.update(action.params)
        child_state.estimated_memory_gb += action.estimated_memory_delta_gb
        child_state.estimated_throughput_tps += action.estimated_throughput_delta_pct

        # Create child node
        valid_actions = self.action_library.get_valid_actions(child_state)
        child = MCTSNode(state=child_state, parent=node, untried_actions=valid_actions)

        node.children[action_name] = child
        return child

    def _select_action_to_try(self, node: MCTSNode) -> str:
        """
        Select which untried action to expand.
        Can incorporate LLM prior or learned heuristics here.
        """
        # Check context-scoped simulation priors.
        prior_scope = self._prior_scope()
        priors = []
        for action_name in node.untried_actions:
            prior = (
                self.knowledge_base.get("action_priors", {})
                .get(prior_scope, {})
                .get(action_name, 0.5)
            )
            priors.append((action_name, prior))

        # Use weighted random selection based on priors
        if priors:
            total = sum(p for _, p in priors)
            r = random.uniform(0, total)
            cumulative = 0
            for action_name, prior in priors:
                cumulative += prior
                if cumulative >= r:
                    return action_name

        # Fallback: random
        return random.choice(node.untried_actions)

    def _evaluate(self, state: OptimizationState, optimization_goal: str) -> float:
        """Evaluate a simulated state and return a bounded score."""
        cache_context = self._cache_context(optimization_goal)
        state_hash = state.get_hash(cache_context)

        # Check cache
        if state_hash in self.knowledge_base.get("evaluations", {}):
            self.cache_hits += 1
            return self.knowledge_base["evaluations"][state_hash]

        self.total_evaluations += 1

        score = self._simulation_score(state, optimization_goal)

        # Cache
        if "evaluations" not in self.knowledge_base:
            self.knowledge_base["evaluations"] = {}
        self.knowledge_base["evaluations"][state_hash] = score

        # Update action priors based on results
        self._update_action_priors(state, score)

        return score

    def _update_action_priors(self, state: OptimizationState, score: float):
        """Update context-scoped simulation priors."""
        if "action_priors" not in self.knowledge_base:
            self.knowledge_base["action_priors"] = {}

        prior_scope = self._prior_scope()
        scoped_priors = self.knowledge_base["action_priors"].setdefault(prior_scope, {})
        for action in state.applied_actions:
            current = scoped_priors.get(action, 0.5)
            # Exponential moving average
            scoped_priors[action] = 0.9 * current + 0.1 * score

    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate score up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += score
            node = node.parent

    def _simulation_score(
        self,
        state: OptimizationState,
        optimization_goal: str,
    ) -> float:
        """Return a bounded heuristic score that cannot be read as a benchmark."""
        gpu_memory_gb = float(self.hardware.get("gpu_memory_gb", 80))
        if state.estimated_memory_gb > gpu_memory_gb:
            return 0.0

        throughput_score = max(
            0.0,
            min(1.0, (state.estimated_throughput_tps + 100.0) / 200.0),
        )
        memory_score = max(
            0.0,
            min(1.0, 1.0 - (state.estimated_memory_gb / gpu_memory_gb)),
        )

        if optimization_goal == "throughput":
            return throughput_score
        if optimization_goal == "memory":
            return memory_score
        return 0.5 * throughput_score + 0.5 * memory_score

    def _cache_context(self, optimization_goal: str) -> dict[str, Any]:
        """Build the complete identity for a cached simulation evaluation."""
        return {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "evaluation_mode": self.evaluation_mode,
            "optimization_goal": optimization_goal,
            "hardware": self.hardware,
            "model": self.model,
            "evaluation_context": self.evaluation_context,
        }

    def _prior_scope(self) -> str:
        """Return a context hash for action priors."""
        payload = {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "evaluation_mode": self.evaluation_mode,
            "hardware": self.hardware,
            "model": self.model,
            "evaluation_context": self.evaluation_context,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _get_avg_depth(self, node: MCTSNode, depth: int = 0) -> float:
        """Calculate average depth of the tree."""
        if not node.children:
            return depth

        child_depths = [self._get_avg_depth(c, depth + 1) for c in node.children.values()]
        return sum(child_depths) / len(child_depths)

    @staticmethod
    def _node_depth(node: MCTSNode) -> int:
        """Return the depth of a node from the root."""
        depth = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def _generate_recommendations(self, node: MCTSNode) -> list[str]:
        """Generate human-readable recommendations from best configuration."""
        recommendations = []

        for action_name in node.state.applied_actions:
            action = self.action_library.actions.get(action_name)
            if action:
                domain = action.domain.value
                params = action.params
                recommendations.append(f"[{domain.upper()}] Apply {action_name}: {params}")

        return recommendations

    def _load_knowledge_base(self) -> dict[str, Any]:
        """Load the versioned simulation knowledge base."""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid MCTS knowledge base: {self.knowledge_base_path}"
                ) from exc
            if data.get("schema_version") == KNOWLEDGE_SCHEMA_VERSION:
                return data
        return {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "evaluations": {},
            "action_priors": {},
        }

    def _save_knowledge_base(self):
        """Save the simulation knowledge base."""
        with open(self.knowledge_base_path, "w") as f:
            json.dump(self.knowledge_base, f, indent=2, sort_keys=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def search_optimal_config(
    model_config: dict[str, Any],
    hardware_config: dict[str, Any],
    optimization_goal: str = "throughput",
    budget: int = 100,
    simulation: bool = False,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search simulated configurations after an explicit opt in.

    Args:
        model_config: Model configuration
        hardware_config: Hardware configuration
        optimization_goal: "throughput", "memory", or "balanced"
        budget: Search budget (number of rollouts)

    Returns:
        Highest-scoring simulated configuration. This is not benchmark evidence.
    """
    optimizer = MCTSOptimizer(
        hardware_config,
        model_config,
        simulation=simulation,
        evaluation_context=evaluation_context,
    )
    return optimizer.search(
        budget=budget,
        optimization_goal=optimization_goal,
        verbose=False,
    )


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCTS Optimization Search")
    parser.add_argument("--model-size", type=float, default=70, help="Model size in billions")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=float, default=80, help="GPU memory in GB")
    parser.add_argument("--gpu-arch", default="hopper", help="GPU architecture")
    parser.add_argument(
        "--goal", default="throughput", choices=["throughput", "memory", "balanced"]
    )
    parser.add_argument("--budget", type=int, default=100, help="Search budget")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Acknowledge that results are heuristic and not benchmark evidence",
    )

    args = parser.parse_args()
    if not args.simulate:
        parser.error(
            "MCTS is simulation-only. Pass --simulate to explore heuristics, or "
            "use the campaign executor for measured experiments."
        )

    model_config = {
        "parameters_billions": args.model_size,
        "num_layers": int(args.model_size * 1.2),  # Rough estimate
        "hidden_size": int((args.model_size * 1e9 / 100) ** 0.5 * 128),
    }

    hardware_config = {
        "num_gpus": args.num_gpus,
        "gpu_memory_gb": args.gpu_memory,
        "gpu_arch": args.gpu_arch,
        "has_nvlink": True,
    }

    print("\n🎯 MCTS Optimization Search")
    print(f"   Model: {args.model_size}B parameters")
    print(f"   Hardware: {args.num_gpus}x {args.gpu_arch} ({args.gpu_memory}GB each)")
    print(f"   Goal: {args.goal}")
    print(f"   Budget: {args.budget} rollouts")
    print()

    print("   Mode: SIMULATION ONLY. Results are not performance evidence.")

    optimizer = MCTSOptimizer(
        hardware_config,
        model_config,
        simulation=True,
    )
    result = optimizer.search(
        budget=args.budget,
        optimization_goal=args.goal,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("HIGHEST-SCORING SIMULATED CONFIGURATION")
    print("=" * 60)
    print(f"\nScore: {result['best_score']:.4f}")
    print(f"Simulated Throughput Delta: {result['simulated_throughput_delta_pct']:+.1f}%")
    print(f"Simulated Memory Delta: {result['simulated_memory_delta_gb']:+.1f} GB")

    print("\n📋 Applied Optimizations:")
    for action in result["best_actions"]:
        print(f"   • {action}")

    print("\n⚙️  Configuration:")
    print(json.dumps(result["best_config"], indent=4))

    print("\n📊 Search Statistics:")
    stats = result["search_statistics"]
    print(f"   Iterations: {stats['total_iterations']}")
    print(f"   Evaluations: {stats['total_evaluations']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   Time: {stats['search_time_seconds']:.2f}s")

    print("\n💡 Recommendations:")
    for rec in result["recommendations"]:
        print(f"   {rec}")


if __name__ == "__main__":
    main()
