#!/usr/bin/env python3
"""CLI entry point for optimization search.

Usage:
    python -m core.optimization.search mcts --simulate --model-size 70 --goal throughput
    python -m core.optimization.search oracle suggest --model-size 70
    python -m core.optimization.search oracle ask "Why is my attention slow?"
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Optimization Search: MCTS simulator and LLM oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MCTS command
    mcts_parser = subparsers.add_parser("mcts", help="Run MCTS optimization search")
    mcts_parser.add_argument("--model-size", type=float, default=70, help="Model size in billions")
    mcts_parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    mcts_parser.add_argument("--gpu-memory", type=float, default=80, help="GPU memory in GB")
    mcts_parser.add_argument("--gpu-arch", default="hopper", help="GPU architecture")
    mcts_parser.add_argument(
        "--goal", default="throughput", choices=["throughput", "memory", "balanced"]
    )
    mcts_parser.add_argument("--budget", type=int, default=100, help="Search budget")
    mcts_parser.add_argument("--verbose", "-v", action="store_true")
    mcts_parser.add_argument("--json", action="store_true", help="Output as JSON")
    mcts_parser.add_argument(
        "--simulate",
        action="store_true",
        help="Acknowledge that results are heuristic and not benchmark evidence",
    )

    # Oracle command
    oracle_parser = subparsers.add_parser("oracle", help="Query the LLM optimization oracle")
    oracle_subparsers = oracle_parser.add_subparsers(dest="oracle_command")

    # Oracle suggest
    suggest_parser = oracle_subparsers.add_parser("suggest", help="Get optimization suggestions")
    suggest_parser.add_argument("--model-size", type=float, help="Model size in billions")
    suggest_parser.add_argument("--profile", help="Path to profile JSON")
    suggest_parser.add_argument("--num", type=int, default=5, help="Number of suggestions")
    suggest_parser.add_argument("--json", action="store_true")

    # Oracle ask
    ask_parser = oracle_subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", nargs="+", help="Your question")

    # Oracle validate
    validate_parser = oracle_subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config", help="Path to config JSON")
    validate_parser.add_argument("--model-size", type=float, default=70)

    args = parser.parse_args()

    if args.command == "mcts":
        from .mcts_optimizer import MCTSOptimizer

        if not args.simulate:
            mcts_parser.error(
                "MCTS is simulation-only. Pass --simulate to explore heuristics, "
                "or use the campaign executor for measured experiments."
            )

        model_config = {
            "parameters_billions": args.model_size,
            "num_layers": int(args.model_size * 1.2),
            "hidden_size": int((args.model_size * 1e9 / 100) ** 0.5 * 128),
        }

        hardware_config = {
            "num_gpus": args.num_gpus,
            "gpu_memory_gb": args.gpu_memory,
            "gpu_arch": args.gpu_arch,
            "has_nvlink": True,
        }

        if not args.json:
            print("\nMCTS HEURISTIC SIMULATION")
            print("   Results are not performance evidence.")
            print(f"   Model: {args.model_size}B parameters")
            print(f"   Hardware: {args.num_gpus}x {args.gpu_arch} ({args.gpu_memory}GB each)")
            print(f"   Goal: {args.goal}")
            print(f"   Budget: {args.budget} rollouts\n")

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

        if args.json:
            print(json.dumps(result, indent=2))
        else:
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

    elif args.command == "oracle":
        from .llm_oracle import LLMOracle

        oracle = LLMOracle()

        if args.oracle_command == "suggest":
            model_config = {"parameters_billions": args.model_size} if args.model_size else None

            profile_data = {}
            if args.profile:
                from pathlib import Path

                if Path(args.profile).exists():
                    with open(args.profile) as f:
                        profile_data = json.load(f)

            suggestions = oracle.suggest_optimizations(
                model_config=model_config,
                profile_data=profile_data,
                num_suggestions=args.num,
            )

            if args.json:
                print(json.dumps([s.to_dict() for s in suggestions], indent=2))
            else:
                print("\n🔮 Optimization Suggestions")
                print("=" * 60)
                for i, s in enumerate(suggestions, 1):
                    print(f"\n{i}. {s.title}")
                    print(f"   {s.description}")
                    print(
                        f"   Expected: {s.expected_speedup} speedup, {s.expected_memory_impact} memory"
                    )
                    print(f"   Difficulty: {s.difficulty} | Category: {s.category}")
                    if s.implementation_steps:
                        print("   Steps:")
                        for step in s.implementation_steps[:3]:
                            print(f"     • {step}")

        elif args.oracle_command == "ask":
            question = " ".join(args.question)
            print(f"\n🔮 Question: {question}\n")
            answer = oracle.ask(question)
            print(answer)

        elif args.oracle_command == "validate":
            with open(args.config) as f:
                config = json.load(f)

            result = oracle.validate_config(
                config=config.get("optimization", config),
                model_config={"parameters_billions": args.model_size},
            )

            print("\n🔮 Configuration Validation")
            print("=" * 60)
            print(f"Valid: {'✅' if result['valid'] else '❌'}")

            if result["issues"]:
                print("\nIssues:")
                for issue in result["issues"]:
                    print(f"  ❌ {issue}")

            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  ⚠️ {warning}")

            if result["recommendations"]:
                print("\nRecommendations:")
                for rec in result["recommendations"]:
                    print(f"  💡 {rec}")

        else:
            oracle_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
