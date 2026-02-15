import sys
import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
import csv


from Problem import Problem
from s348365 import solution
from src.solver import solve
from .utils import fix_edge



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
    1. Checks if total gold collected equals the city's gold.
    2. FIXES tiny floating-point errors (< 0.001) in-place.
    3. Prints errors only for major logic bugs (missing chunks).
    """

    collected_totals = defaultdict(float)
    for node_id, gold_amount in path:
        if node_id != 0:
            collected_totals[node_id] += gold_amount

    # check against the problem definition
    for n in problem.graph.nodes:
        if n == 0: continue
        
        real_gold = problem.graph.nodes[n]["gold"]
        total_picked = collected_totals[n]
        
        # TO COMPARE FLOATS use math.isclose or a small tolerance
        if not abs(real_gold - total_picked) < 1e-4:
            print(f"ERROR at City {n}: Expected {real_gold}, got {total_picked}")
            import sys
            sys.exit(1)
            
    # clean up final path --> at this point only very small errors should be present, so we can fix them in-place
    final_path = []
    for node, gold in path:
        # Round the gold pickup to the desired precision
        # This removes the 778.6051135766882 mess
        clean_gold = round(gold, 2)
        final_path.append((node, clean_gold))
    return final_path

def debug_path_feasibility(path, problem):
    """
    Strictly checks that every step is a valid DIRECT EDGE.
    Uses has_edge() which is O(1) (instant), unlike has_path().
    """
    G = problem.graph
    current = 0 # start at depot
    valid_path = []

    for step, (next_node, gold) in enumerate(path):
        if current == next_node:
            valid_path.append((current, gold))
        elif G.has_edge(current, next_node):
            valid_path.append((next_node, gold))
        else:
            segment = fix_edge(problem, current, next_node, gold)
            valid_path.extend(segment)
        
        current = next_node
        
    return valid_path


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
    # baseline_path = problem.baseline()
    t_baseline = time.time() - t0
    baseline_cost = problem.baseline()

    # --------------------------------------------------
    # ILS ONLY
    # --------------------------------------------------
    t0 = time.time()
    ils_path = solution(problem)

    ils_path = debug_gold_conservation_path(ils_path, problem)
    final_path = debug_path_feasibility(ils_path, problem)
    t_ils = time.time() - t0

    ils_cost = compute_solution_cost(problem, final_path)
    improvement = 100.0 * (baseline_cost - ils_cost) / baseline_cost
    print(f"ILS:      cost={ils_cost:.2f}, impr={improvement:.2f}%")
    
    # ------------------------------------------------------------
    # ILS + LNS (only meaningful for larger instances, >50 cities)
    # ------------------------------------------------------------
    num_cities = len(problem.graph.nodes) - 1
    time_needed = t_ils
    if num_cities >= 50:
        t0 = time.time()
        lns_path = solution(problem)
        final_path = debug_gold_conservation_path(lns_path,problem)
        final_path = debug_path_feasibility(final_path,problem)
        t_lns = time.time() - t0
        lns_cost = compute_solution_cost(problem, final_path)
        improvement = 100.0 * (baseline_cost - lns_cost) / baseline_cost
        print(
            f"ILS+LNS:  cost={lns_cost:.2f}, impr={improvement:.2f}%"
        )
        time_needed += t_lns

    print(final_path)

    test_results.append({
        "n_cities": n,
        "density": density,
        "alpha": alpha,
        "beta": beta,
        "baseline_cost": baseline_cost,
        "ils_cost": ils_cost,
        "improvement": improvement
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

    time_needed = {}
    for n in n_cities:
        for density in density_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    t0 = time.time()
                    results.append(run_test(n, density, alpha, beta, seed))
                    t1 = time.time()
                    config = (n, density, alpha, beta)
                    time_needed[config] = t1 - t0
                    
                    print(f"----- TIME NEEDED: {t1 - t0:.4f} seconds -----\n")

    flat_results = [r for sublist in results for r in sublist] # for easier printing
    export_summary_csv(flat_results, ("beta",))
    export_summary_csv(flat_results, ("alpha",))
    export_summary_csv(flat_results, ("n_cities",))
    export_summary_csv(flat_results, ("density",))
    export_summary_csv(flat_results, ("n_cities", "alpha", "beta"))

    with open('time_needed_test.csv', 'w', newline='') as csvfile:
        fieldnames = ['n_cities', 'density', 'alpha', 'beta', 'time_to_run']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config, time_X in time_needed.items():
            row = dict(zip(fieldnames[:-1], config))
            row['time_to_run'] = f"{time_X:.4f}"
            writer.writerow(row)

