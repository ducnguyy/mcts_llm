import time
import argparse
import json

import llm_client
from tasks import get_task, get_all_tasks

# Algorithm imports
from mctsr import mctsr
from uct_answer import uct_answer
from nmcs_answer import nmcs_answer
from nrpa_answer import nrpa_answer
from sh_answer import sh_answer
from puct_answer import puct_answer
from baselines_answer import single_shot, best_of_n, self_refine_chain

ALGORITHMS = {
    "Single-shot": {
        "fn": lambda p, q, t, i: single_shot(p, q, t, i),
        "type": "baseline",
    },
    "Best-of-4": {
        "fn": lambda p, q, t, i: best_of_n(p, q, t, i, n=4),
        "type": "baseline",
    },
    "SelfRefine-4": {
        "fn": lambda p, q, t, i: self_refine_chain(p, q, t, i, k=4),
        "type": "baseline",
    },

    "MCTSr": {
        "fn": lambda p, q, t, i: mctsr(p, q, t, i, max_iter=8),
        "type": "paper",
    },
    
    "UCT": {
        "fn": lambda p, q, t, i: uct_answer(p, q, t, i, n_simulations=8),
        "type": "course",
    },
    "NMCS-1": {
        "fn": lambda p, q, t, i: nmcs_answer(p, q, t, i, level=1, max_refinements=3),
        "type": "course",
    },
    "NRPA-1": {
        "fn": lambda p, q, t, i: nrpa_answer(p, q, t, i, level=1, n_iter=4),
        "type": "course",
    },
    "SH": {
        "fn": lambda p, q, t, i: sh_answer(p, q, t, i, n_candidates=4),
        "type": "course",
    },
    "PUCT": {
        "fn": lambda p, q, t, i: puct_answer(p, q, t, i, n_simulations=8),
        "type": "course",
    },
}


def run_single(algo_name, algo_fn, task, instance):
    """Run one algorithm on one instance, return results dict."""
    prompt = task.get_prompt(instance)
    question = prompt  # use the full prompt as question for self-eval

    start = time.time()
    try:
        answer, ext_reward, info = algo_fn(prompt, question, task, instance)
        elapsed = time.time() - start
        status = "OK"
    except Exception as e:
        elapsed = time.time() - start
        answer = f"ERROR: {str(e)}"
        ext_reward = 0.0
        info = {"error": str(e)}
        status = "ERROR"

    # Model-based reward
    if status == "OK":
        try:
            raw = llm_client.self_evaluate(question, answer)
            model_reward = llm_client.normalize_score(raw)
        except Exception:
            model_reward = 0.0
    else:
        model_reward = 0.0

    result = {
        "algorithm": algo_name,
        "task": task.name,
        "instance_id": instance.get("id", 0),
        "answer": answer[:500],  
        "external_reward": ext_reward,
        "model_reward": round(model_reward, 3),
        "time": round(elapsed, 2),
        "status": status,
        **{k: v for k, v in info.items() if k != "algorithm"},  
    }

    # Add task-specific details
    if hasattr(task, 'format_result'):
        result["details"] = task.format_result(instance, answer)

    return result


def main():
    parser = argparse.ArgumentParser(description="MCTSr evaluation harness")
    parser.add_argument("--task", type=str, default=None,
                        help="Run only this task (haiku, chess, or game24)")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="Run only these algorithms")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 2 instances per task")
    parser.add_argument("--n_instances", type=int, default=None,
                        help="Number of instances per task")
    parser.add_argument("--output", type=str, default="results_mctsr.json",
                        help="Output file")
    args = parser.parse_args()

    # Select tasks
    if args.task:
        tasks = [get_task(args.task)]
    else:
        tasks = get_all_tasks()

    # Select algorithms
    if args.algorithms:
        algo_items = [(k, v) for k, v in ALGORITHMS.items()
                      if k in args.algorithms]
    else:
        algo_items = list(ALGORITHMS.items())

    # Number of instances
    n_instances = args.n_instances
    if args.quick:
        n_instances = 2

    all_results = []

    for task in tasks:
        instances = task.get_instances()
        if n_instances:
            instances = instances[:n_instances]

        print(f"\n{'='*70}")
        print(f"TASK: {task.name} ({len(instances)} instances)")
        print(f"{'='*70}")

        for inst in instances:
            inst_desc = inst.get("topic", inst.get("description", f"#{inst.get('id', '?')}"))
            print(f"\n  Instance: {inst_desc}")
            print(f"  {'-'*60}")

            for algo_name, algo_info in algo_items:
                print(f"    Running {algo_name}...", end=" ", flush=True)
                result = run_single(algo_name, algo_info["fn"], task, inst)
                all_results.append(result)

                status_icon = "✓" if result["external_reward"] > 0 else "✗"
                print(f"{status_icon} ext={result['external_reward']:.2f} "
                      f"| model={result['model_reward']:.2f} "
                      f"| calls={result.get('llm_calls', '?')} "
                      f"| {result['time']}s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print_summary(all_results, algo_items, tasks)


def print_summary(all_results, algo_items, tasks):
    """Print a summary table."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for task in tasks:
        print(f"\n--- Task: {task.name} ---")
        print(f"{'Algorithm':<16} {'Type':<8} {'AvgExt':>8} {'AvgModel':>9} {'Sat.Rate':>9} {'Avg Calls':>10} {'Avg Time':>10}")
        print("-" * 75)

        for algo_name, algo_info in algo_items:
            results = [r for r in all_results
                       if r["algorithm"] == algo_name and r["task"] == task.name]
            if not results:
                continue

            avg_ext = sum(r["external_reward"] for r in results) / len(results)
            avg_model = sum(r.get("model_reward", 0.0) for r in results) / len(results)
            sat_rate = sum(1 for r in results if r["external_reward"] > 0) / len(results)
            avg_calls = sum(r.get("llm_calls", 0) for r in results) / len(results)
            avg_time = sum(r["time"] for r in results) / len(results)

            type_label = algo_info["type"]
            print(f"{algo_name:<16} {type_label:<8} {avg_ext:>8.3f} {avg_model:>9.3f} {sat_rate:>8.0%} {avg_calls:>10.0f} {avg_time:>9.1f}s")


if __name__ == "__main__":
    main()