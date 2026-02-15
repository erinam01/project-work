import random
from collections import Counter
import networkx as nx

def get_precision(number):
    """Counts decimal places of a number to determine necessary precision."""
    s = str(number)
    if '.' not in s:
        return 0
    return len(s) - s.index('.') - 1

def split_cities_by_max_load(problem, city_max_loads):
    """
    Convert each city into one or more "virtual nodes" representing partial pickups
    returns: list of virtual nodes: [(city_id, pickup_amount), ...]; --> several cities with the same id's will appear.
    Meaning that the algorithm decided that city had far too much gold and it needed to be split for efficient cos solution
    """
    virtual_nodes = []

    # precision handling: determine max decimal places across all gold values to set a consistent rounding precision for virtual nodes
    all_gold = [problem.graph.nodes[n]["gold"] for n in problem.graph.nodes]
    max_precision = 0
    for g in all_gold:
        p = get_precision(g)
        if p > max_precision:
            max_precision = p
            
    # add a safety buffer of 1 decimal
    precision = max(2, max_precision)

    for node in problem.graph.nodes:
        if node == 0: continue 

        remaining = problem.graph.nodes[node]["gold"]
        limit = city_max_loads.get(node, remaining)
        
        # sanity check to avoid infinite loops if limit is effectively 0
        if limit < 1e-6: limit = 1.0 

        while remaining > 0:
            if remaining <= limit:
                pickup = remaining
            else:
                pickup = limit
            
            # round to fix precision
            pickup = round(pickup, precision)
            
            remaining -= pickup
            remaining = round(remaining, precision)

            virtual_nodes.append((node, pickup))
    return virtual_nodes

def routes_cost(routes, problem, shortest):
    total = 0.0

    for route in routes:
        current = 0 
        load = 0.0

        for node in route:
            city_id, pickup = node 
            
            # DO NOT READ 'gold' FROM problem.graph
            # WRONG: remaining_gold = problem.graph.nodes[city_id]["gold"]
            # through virtual nodes logic, the only gold we need to lookup is the one associated to the city
            # not to the problem
            
            dist = shortest[current][city_id]
            total += dist + (problem.alpha * dist * load) ** problem.beta

            load += pickup
            current = city_id

        dist = shortest[current][0]
        total += dist + (problem.alpha * dist * load) ** problem.beta

    return total


def fix_edge(problem, u, v, v_gold):
    """
    Checks if the move u -> v is valid.
    - If valid (direct edge exists), returns [(v, v_gold)].
    - If invalid (no edge), returns the shortest path sequence: 
      [(stop1, 0.0), (stop2, 0.0), ... (v, v_gold)]
    """
    G = problem.graph
    
    # if nodes are the same, no move needed (but we might need to pick up gold)
    if u == v:
        return [(v, v_gold)]

    # check if direct edge exists
    if G.has_edge(u, v):
        return [(v, v_gold)]
    
    # if no edge, find shortest path (the "fix")
    # get path: [u, stop1, stop2, ..., v]
    path_nodes = nx.shortest_path(G, source=u, target=v, weight='dist')
    
    fixed_segment = []
    
    # add intermediate stops (0.0 gold)
    # w slice [1:-1] to skip 'u' (start) and 'v' (end)
    for node in path_nodes[1:-1]:
        fixed_segment.append((node, 0.0))
        
    # add final destination with its original gold pickup
    fixed_segment.append((v, v_gold))
    
    return fixed_segment


def build_path_from_routes(routes, problem, shortest):
  # convert virtual node routes into final path [(city, gold), ...]
  # takes the hierarchical solution structure used by the optimizer (a List of Lists of virtual nodes) 
  # and flattens it into a single, linear sequence (a List of Tuples), adding return to depot as well
    path = [(0,0.0)] # start at depot
    current_node = 0 
    
    for route in routes:
        for node in route:
            target_id, pickup = node
            segment = fix_edge(problem, current_node, target_id, pickup)
            path.extend(segment)
            
            current_node = target_id
            
        # return to depot
        if current_node != 0:
            segment = fix_edge(problem, current_node, 0, 0.0)
            path.extend(segment)
            current_node = 0

    return path


# -------- NEIGHBOURHOOD OPERATORS --------
def move_city_virtual(routes):
    routes = [r[:] for r in routes if r]
    if len(routes) < 2:
        return routes

    # randomly
    r1, r2 = random.sample(range(len(routes)), 2)
    node = random.choice(routes[r1])
    routes[r1].remove(node)
    routes[r2].append(node)

    if not routes[r1]:
        del routes[r1]

    return routes

def swap_cities_virtual(routes):
    routes = [r[:] for r in routes if len(r) > 1]
    if not routes:
        return routes

    r = random.choice(routes)
    i, j = random.sample(range(len(r)), 2)
    r[i], r[j] = r[j], r[i]
    return routes

def split_route_virtual(routes):
    routes = [r[:] for r in routes if len(r) > 2]
    if not routes:
        return routes

    r = random.choice(routes)
    cut = random.randint(1, len(r) - 1)
    routes.remove(r)
    routes.append(r[:cut])
    routes.append(r[cut:])
    return routes 

def swap_between_routes_virtual(routes):
    """Swaps a node from route A with a node from route B"""
    routes = [r[:] for r in routes if r]
    if len(routes) < 2: return routes
    r1_idx, r2_idx = random.sample(range(len(routes)), 2)
    r1, r2 = routes[r1_idx], routes[r2_idx]
    if not r1 or not r2: return routes
    i = random.randint(0, len(r1) - 1)
    j = random.randint(0, len(r2) - 1)
    r1[i], r2[j] = r2[j], r1[i]
    return routes

def merge_routes_virtual(routes):
    """Merges two random routes into one"""
    routes = [r[:] for r in routes if r]
    if len(routes) < 2: return routes
    r1_idx, r2_idx = random.sample(range(len(routes)), 2)
    routes[r1_idx].extend(routes[r2_idx])
    del routes[r2_idx]
    return routes

def repair_routes(routes, virtual_nodes):
    """
    Ensure all virtual nodes are present, respecting duplicates; i call thse nodes '''virtual''' because many '''duplicates''' are present:
    of course they are not really duplicates but rather different chunks of the same city with different amounts of gold (their sum should be 
    equal to the tot amount of gold)
    ils and lns operators may cause to drop or destroy parts of a solution; this solution checks for FEASIBILITY
    ROUTES: current working solution (a list of lists)
    VIRTUAL NODES: "Master List" of every single gold chunk that exists in the problem
    """
    # count how many chunks we should have, ie, how many times the city has been split
    virtual_nodes = [(v[0], round(v[1], 6)) for v in virtual_nodes]
    required_counts = Counter(virtual_nodes)

    # COUNTER FROM COLLECTIONS LOGIC WAS ADDED --> because of artificial duplicates logic, we might have the same city appearing multiple times 
    # with identical or different gold amounts

    # count how many of each chunk we CURRENTLY have in the routes --> how many chunks this route asks us to take
    # (for efficiency reasons that the route counted)
    # normalize current routes before counting
    normalized_routes = []
    for r in routes:
        normalized_routes.append([(vn[0], round(vn[1], 6)) for vn in r])

    current_nodes = [vn for r in normalized_routes for vn in r]
    current_counts = Counter(current_nodes)

    missing = []

    # find missing nodes, either artificial or real, that don't appear in our solution but appear in the problem definition, and so we add them
    for vn, count in required_counts.items():
        current_have = current_counts[vn]
        if current_have < count:
            # we are missing (count - current_have) copies of this specific chunk
            diff = count - current_have
            # if we are supposed to have two chunks of (5, 50.0) but we only have one in our current routes, diff becomes 1
            missing.extend([vn] * diff)

    # append missing chunks as new individual routes
    # (the optimizer will later merge these into better routes)
    for vn in missing:
        routes.append([vn])

    # NOTE FOR INEFFICIENCY: if one part of the code generates 33.33333333 and another generates 33.33333334 due to tiny floating-point math differences,
    # Counter will treat them as different items !! --> might be the cause of some inefficient solutions

    # NOTE: this works also for cities that haven't been split, or that have no 'virtual node' logic, ie, cities COUNT=1; handles them the same

    return routes
