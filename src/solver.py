import random
import networkx as nx
from .utils import (
    build_path_from_routes,
    routes_cost,
    repair_routes,
    move_city_virtual,
    swap_cities_virtual,
    split_route_virtual, 
    swap_between_routes_virtual,
    merge_routes_virtual,
    split_cities_by_max_load
)
from .lns import lns_virtual

def compute_city_max_loads(problem, beta=None):
    """
    Returns a dictionary {city_id: max_load} adaptive to parameters.
    Replaces the old 'compute_max_load' that returned a single float.
    """
    if beta is None: beta = problem.beta
    alpha = problem.alpha
    
    nodes = problem.graph.nodes
    # Filter out depot (0)
    gold_values = [nodes[n]["gold"] for n in nodes if n != 0]
    
    if not gold_values: return {}

    # 1. Compute Global Physics Limit (The Ceiling)
    if alpha == 0:
        # If alpha is 0, physics doesn't matter -> Infinite capacity
        global_cap = sum(gold_values)
    else:
        num_nodes = len(nodes)
        max_gold = max(gold_values)
        
        # Scaling factors based on graph size
        if num_nodes <= 20: p = 0.5   
        elif num_nodes <= 50: p = 0.7 
        else: p = 0.9    

        if 1.5 <= beta <= 2.5:
            p += 0.3             
        
        # Physics drag calculation
        f_beta = 1.0 / (beta ** p)
        scale = min(1.0, 50 / num_nodes)
        
        global_cap = max(1.0, max_gold * f_beta * scale)

    # 2. Build the City-Specific Dictionary
    city_max_load = {}
    for n in nodes:
        if n == 0: continue
        # The limit is the smaller of: Global Cap OR The City's total gold
        city_max_load[n] = min(nodes[n]["gold"], global_cap)
        
    return city_max_load

def ils(problem, shortest, max_iter=300, city_max_load=None):
    if city_max_load is None:
        city_max_load = compute_city_max_loads(problem)

    virtual_nodes = split_cities_by_max_load(problem, city_max_load)

    # --- Initialization: Best of Random vs Star ---
    # 1. Random (One big tour)
    vn_shuffled = virtual_nodes[:]
    random.shuffle(vn_shuffled)
    route_random = [vn_shuffled]
    
    # 2. Star (Baseline)
    route_star = [[vn] for vn in virtual_nodes]

    # pick the best start to avoid negative improvement
    if routes_cost(route_star, problem, shortest) < routes_cost(route_random, problem, shortest):
        best_routes = route_star
    else:
        best_routes = route_random
        
    best_cost = routes_cost(best_routes, problem, shortest)

    operators = [
        move_city_virtual, 
        swap_cities_virtual, 
        split_route_virtual,
        merge_routes_virtual,
        swap_between_routes_virtual
    ]

    for _ in range(max_iter):
        op = random.choice(operators)
        candidate = op(best_routes)
        candidate = repair_routes(candidate, virtual_nodes)

        cost = routes_cost(candidate, problem, shortest)
        if cost < best_cost:
            best_routes = candidate
            best_cost = cost

    return best_routes

def solve(problem):
    shortest = dict(
        nx.all_pairs_dijkstra_path_length(problem.graph, weight="dist")
    )

    # Always start with ILS
    best_routes = ils(problem, shortest)
    city_max_loads = compute_city_max_loads(problem)

    num_nodes = len(problem.graph.nodes) - 1
    # Optionally refine with LNS
    if num_nodes >= 50:
        best_routes = lns_virtual(
            problem,
            shortest,
            start_routes=best_routes,
            iterations=80,
            destroy_fraction=0.25,
            city_max_load=city_max_loads
        )

    return build_path_from_routes(best_routes, problem, shortest)