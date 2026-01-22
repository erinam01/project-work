import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
import statistics
import csv


from Problem import Problem
from s349370 import Solver
from src.solver import solve

results = []


def compute_solution_cost(problem, path):
    # compute the cost of the solution as per problem definition
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
    baseline_path = solve(problem)
    t_baseline = time.time() - t0
    baseline_cost = compute_solution_cost(problem, baseline_path)



    # --------------------------------------------------
    # ILS ONLY
    # --------------------------------------------------
    pw_ils = Solver(problem)
    t0 = time.time()
    ils_path = pw_ils.solution(use_lns=False)
    t_ils = time.time() - t0
    ils_cost = compute_solution_cost(problem, ils_path)

    impr_ils = 100.0 * (baseline_cost - ils_cost) / baseline_cost

    #print(
    #    f"ILS:      cost={ils_cost:.2f}, time={t_ils:.2f}s, impr={impr_ils:.2f}%"
    #)

    print(f"{ils_path}\n")

    # ------------------------------------------------------------
    # ILS + LNS (only meaningful for larger instances, >50 cities)
    # ------------------------------------------------------------
    num_cities = len(problem.graph.nodes) - 1

    if num_cities >= 50:
        pw_lns = Solver(problem)
        t0 = time.time()
        lns_path = pw_lns.solution(use_lns=True)
        t_lns = time.time() - t0
        lns_cost = compute_solution_cost(problem, lns_path)

        impr_lns = 100.0 * (baseline_cost - lns_cost) / baseline_cost

        #print(
        #    f"ILS+LNS:  cost={lns_cost:.2f}, time={t_lns:.2f}s, impr={impr_lns:.2f}%"
        #)
    
    row = {
        "N": num_cities,
        "density": density,
        "alpha": alpha,
        "beta": beta,
        "baseline_cost": baseline_cost,
        "ils_cost": ils_cost,
        "ils_time": t_ils,
        "ils_impr_pct": impr_ils,
    }

    if num_cities >= 50:
        row.update({
            "lns_cost": lns_cost,
            "lns_time": t_lns,
            "lns_impr_pct": impr_lns,
        })

    results.append(row)




def export_summary_by_alpha(results, filename="summary_by_alpha.csv"):
    bucket = defaultdict(list)

    for r in results:
        if "lns_impr_pct" in r:
            bucket[r["alpha"]].append(r["lns_impr_pct"])
        else:
            bucket[r["alpha"]].append(r["ils_impr_pct"])

    rows = []
    for alpha in sorted(bucket):
        vals = bucket[alpha]
        rows.append({
            "alpha": alpha,
            "mean_improvement_pct": statistics.mean(vals),
            "median_improvement_pct": statistics.median(vals),
            "num_instances": len(vals),
        })

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

def export_summary_by_beta(results, filename="summary_by_beta.csv"):
    bucket = defaultdict(list)

    for r in results:
        if "lns_impr_pct" in r:
            bucket[r["beta"]].append(r["lns_impr_pct"])
        else:
            bucket[r["beta"]].append(r["ils_impr_pct"])

    rows = []
    for beta in sorted(bucket):
        vals = bucket[beta]
        rows.append({
            "beta": beta,
            "mean_improvement_pct": statistics.mean(vals),
            "median_improvement_pct": statistics.median(vals),
            "num_instances": len(vals),
        })

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary by beta saved to {filename}")


def export_results_csv(results, filename="comparison_results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

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

    export_results_csv(results)
    export_summary_by_alpha(results)
    export_summary_by_beta(results)
