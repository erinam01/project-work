from collections import Counter
import random
from .utils import routes_cost, repair_routes, split_cities_by_max_load
# --------------------------------------------------
# DESTROY OPERATOR
# --------------------------------------------------

def destroy(routes, fraction):
    """
    Remove a fraction of cities from the current solution.
    In the case of really inefficient routes a move or swap operator may not be enough
    Returns:
      - partial routes
      - list of removed cities
    """
    all_cities = [c for r in routes for c in r] 
    # flatten all cities into a list
    if not all_cities:
        return routes, []

    k = max(1, int(len(all_cities) * fraction)) 
    # k = number of cities to destroy
    removed = random.sample(all_cities, k)
    # sample the k cities to destroy (at least 1)
    to_remove_counts = Counter(removed)
    new_routes = []
    for r in routes:
        new_route = []
        for city in r:
            if to_remove_counts[city] > 0:
                to_remove_counts[city] -= 1
            else:
                new_route.append(city)
        if new_route:
            new_routes.append(new_route)

    return new_routes, removed


# --------------------------------------------------
# GREEDY REPAIR OPERATOR
# --------------------------------------------------
def greedy_repair_virtual(routes, removed, problem, shortest):
    """
    Reinsert removed virtual nodes into routes in the cheapest position.

    Args:
        routes: current list of routes [[(city_id, pickup), ...], ...]
        removed: list of virtual nodes [(city_id, pickup), ...] to reinsert
        problem: Problem instance
        shortest: precomputed shortest distances
    Returns:
        routes with removed nodes reinserted
    """
    all_costs = [routes_cost(routes, problem, shortest) for r in routes]
    # calculate total cost once
    for vn in removed:
        best_cost = float("inf")
        best_position = None  # (route_index, insert_index)

        new_route_cost = routes_cost([[vn]], problem, shortest)
        if new_route_cost < best_cost:
            best_cost = new_route_cost
            best_position = (len(routes), 0)

        # try inserting vn into all existing routes
        for i, route in enumerate(routes):
            original_cost = all_costs[i]
            for pos in range(len(route) + 1):
                trial_route = route[:pos] + [vn] + route[pos:]
     
                new_route_cost = routes_cost([trial_route], problem, shortest)
                delta = new_route_cost - original_cost
                if delta < best_cost:
                    best_cost = delta
                    best_position = (i,pos)

    # apply the best insertion
        i,pos = best_position
        if i == len(routes):
            routes.append([vn])
            all_costs.append(best_cost)
        else:
            routes[i].insert(pos, vn)
            all_costs[i] += best_cost

    return routes



# --------------------------------------------------
# LNS MAIN LOOP
# --------------------------------------------------
def lns_virtual(problem, shortest, start_routes, iterations=100, destroy_fraction=0.25, city_max_load=None):
    # pass the dictionary to get correct virtual nodes
    virtual_nodes = split_cities_by_max_load(problem, city_max_load)

    best_routes = start_routes
     # the best solution found so far without implementing LNS
    best_cost = routes_cost(best_routes, problem, shortest)

    # EXTRA --> # Adaptive loop: Run fewer iterations for massive problems to save time
    if len(virtual_nodes) > 150:
        iterations = max(20, iterations // 2)

    for _ in range(iterations):
        partial, removed = destroy(best_routes, fraction=destroy_fraction)
        
        candidate = greedy_repair_virtual(partial, removed, problem, shortest)
        candidate = repair_routes(candidate, virtual_nodes)

        cost = routes_cost(candidate, problem, shortest)
        if cost < best_cost:
            best_routes = candidate
            best_cost = cost
        # if randomly deleting some cities from the total trip and them greedily reinserting them improved solution
        # then keep it


    return best_routes

