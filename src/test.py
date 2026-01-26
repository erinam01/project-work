import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
import csv


from Problem import Problem
from s348365 import solution
from src.solver import solve



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

def debug_gold_conservation_path(path, problem):
    """
    Check that total gold collected from each city equals the city's gold.
    Works on final path: [(node, gold), ...]
    """
    collected = {n: 0.0 for n in problem.graph.nodes if n != 0}

    for node, gold in path:
        if node != 0:
            collected[node] += gold

    ok = True
    for n, collected_gold in collected.items():
        real_gold = problem.graph.nodes[n]["gold"]
        if abs(collected_gold - real_gold) > 1e-6:
            print(
                f"Gold mismatch at city {n}: "
                f"collected={collected_gold:.6f}, real={real_gold:.6f}"
            )
            ok = False


def debug_path_feasibility(path, problem):
    """
    Verify that every movement in the solution path is feasible in the graph.
    """
    G = problem.graph
    current = 0

    for step, (next_node, _) in enumerate(path):
        if not nx.has_path(G, current, next_node):
            raise ValueError(
                f"Infeasible path at step {step}: "
                f"{current} → {next_node} does not exist in graph"
            )
        current = next_node


def run_test(num_cities, density, alpha, beta, seed):
    test_results = []

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
    baseline_path = solve(problem)
    t_baseline = time.time() - t0
    baseline_cost = compute_solution_cost(problem, baseline_path)

    # --------------------------------------------------
    # ILS ONLY
    # --------------------------------------------------
    t0 = time.time()
    ils_path = solution(problem)
    t_ils = time.time() - t0
    ils_cost = compute_solution_cost(problem, ils_path)

    impr_ils = 100.0 * (baseline_cost - ils_cost) / baseline_cost

    #print(
    #    f"ILS:      cost={ils_cost:.2f}, time={t_ils:.2f}s, impr={impr_ils:.2f}%"
    #)

    print(f"{ils_path}\n")
    
    debug_gold_conservation_path(ils_path, problem)
    debug_path_feasibility(ils_path, problem)
    # ------------------------------------------------------------
    # ILS + LNS (only meaningful for larger instances, >50 cities)
    # ------------------------------------------------------------
    num_cities = len(problem.graph.nodes) - 1

    if num_cities >= 50:
        t0 = time.time()
        lns_path = solution(problem)
        t_lns = time.time() - t0
        lns_cost = compute_solution_cost(problem, lns_path)

        impr_lns = 100.0 * (baseline_cost - lns_cost) / baseline_cost
        debug_gold_conservation_path(lns_path,problem)
        debug_path_feasibility(lns_path,problem)
        print(
            f"ILS+LNS:  cost={lns_cost:.2f}, time={t_lns:.2f}s, impr={impr_lns:.2f}%"
        )

        #print(
        #    f"ILS+LNS:  cost={lns_cost:.2f}, time={t_lns:.2f}s, impr={impr_lns:.2f}%"
        #)
    
    

    test_results.append({
        "n_cities": n,
        "density": density,
        "alpha": alpha,
        "beta": beta,
        "baseline_cost": baseline_cost,
        "ils_cost": ils_cost,
        "improvement": (baseline_cost - ils_cost) / baseline_cost if baseline_cost > 0 else 0.0,
    })
    return test_results

def export_summary_csv(results, group_keys, filename=None):
    """
    Save aggregated results grouped by group_keys into a CSV file.
    """

    if filename is None:
        filename = "summary_by_" + "_".join(group_keys) + ".csv"

    grouped = defaultdict(list)

    for r in results:
        key = tuple(r[k] for k in group_keys)
        grouped[key].append(r)

    rows = []

    for key, items in grouped.items():
        avg_baseline = sum(i["baseline_cost"] for i in items) / len(items)
        avg_ils = sum(i["ils_cost"] for i in items) / len(items)
        avg_impr = sum(i["improvement"] for i in items) / len(items)

        row = dict(zip(group_keys, key))
        row.update({
            "avg_baseline_cost": avg_baseline,
            "avg_ils_cost": avg_ils,
            "avg_improvement_pct": avg_impr,
            "num_instances": len(items)
        })

        rows.append(row)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)



if __name__ == "__main__":

    n_cities = [10, 50, 100]    
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    beta_values = [0.5, 1, 2, 4]
    density_values = [0.2, 0.5, 1.0]
    seed = 42

    results = []


    for n in n_cities:
        for density in density_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    results.append(run_test(n, density, alpha, beta, seed))
   
    flat_results = [r for sublist in results for r in sublist] # for easier printing
    export_summary_csv(flat_results, ("beta",))
    export_summary_csv(flat_results, ("alpha",))
    export_summary_csv(flat_results, ("n_cities",))
    export_summary_csv(flat_results, ("density",))

    export_summary_csv(flat_results, ("n_cities", "alpha", "beta"))
