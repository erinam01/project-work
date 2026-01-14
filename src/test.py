import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
import csv


from Problem import Problem
from s349370 import Solver

results = []


def compute_solution_cost(problem, path):
    shortest = dict(
        nx.all_pairs_dijkstra_path_length(problem.graph, weight="dist")
    )

    total_cost = 0.0
    current_node = 0
    current_load = 0.0

    for next_node, gold in path:
        dist = shortest[current_node][next_node]
        total_cost += dist + (problem.alpha * dist * current_load) ** problem.beta

        if next_node == 0:
            current_load = 0.0
        else:
            current_load += gold

        current_node = next_node

    return total_cost


def run_test(num_cities, density, alpha, beta, seed):
    print("=" * 70)
    print(
        f"Test config: N={num_cities}, density={density}, "
        f"alpha={alpha}, beta={beta}, seed={seed}"
    )

    random.seed(seed)
    np.random.seed(seed)

    problem = Problem(
        num_cities,
        density=density,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )

    # --------------------------------------------------
    # BASELINE
    # --------------------------------------------------
    t0 = time.time()
    baseline_path = problem.solve()
    t_baseline = time.time() - t0
    baseline_cost = compute_solution_cost(problem, baseline_path)

    print(f"Baseline cost: {baseline_cost:.2f}")

    # --------------------------------------------------
    # ILS ONLY
    # --------------------------------------------------
    pw_ils = Solver(problem)
    t0 = time.time()
    ils_path = pw_ils.solve(use_lns=False)
    t_ils = time.time() - t0
    ils_cost = compute_solution_cost(problem, ils_path)

    impr_ils = 100.0 * (baseline_cost - ils_cost) / baseline_cost

    print(
        f"ILS:      cost={ils_cost:.2f}, time={t_ils:.2f}s, impr={impr_ils:.2f}%"
    )

    # --------------------------------------------------
    # ILS + LNS (only meaningful for larger instances)
    # --------------------------------------------------
    num_cities = len(problem.graph.nodes) - 1

    if num_cities >= 50:
        pw_lns = Solver(problem)
        t0 = time.time()
        lns_path = pw_lns.solve(use_lns=True)
        t_lns = time.time() - t0
        lns_cost = compute_solution_cost(problem, lns_path)

        impr_lns = 100.0 * (baseline_cost - lns_cost) / baseline_cost

        print(
            f"ILS+LNS:  cost={lns_cost:.2f}, time={t_lns:.2f}s, impr={impr_lns:.2f}%"
        )

    print("=" * 70)


def print_summary_by_beta(results):
    summary = defaultdict(list)

    for r in results:
        summary[r["beta"]].append(r["improvement_pct"])

    print("\nSUMMARY: Improvement over baseline by β")
    print("=" * 55)
    print(f"{'β':<6}{'Mean (%)':>12}{'Min (%)':>12}{'Max (%)':>12}")
    print("-" * 55)

    for beta in sorted(summary.keys()):
        values = summary[beta]
        mean_v = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)

        print(f"{beta:<6}{mean_v:>12.2f}{min_v:>12.2f}{max_v:>12.2f}")

def export_results_csv(results, filename="comparison_results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {filename}")

if __name__ == "__main__":

    n_cities = [10, 50, 100]    
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    beta_values = [0.5, 1, 2, 4]
    density_values = [0.2, 0.5, 1.0]
    seed = 42

    for n in n_cities:
        for density in density_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    run_test(n, density, alpha, beta, seed)

    print_summary_by_beta(results)
