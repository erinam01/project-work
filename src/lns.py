import random
from .utils import routes_cost, repair_routes, split_cities_by_max_load


# --------------------------------------------------
# DESTROY OPERATOR
# --------------------------------------------------

def destroy(routes, fraction=0.25):
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
    removed = set(random.sample(all_cities, k)) 
    # sample the k cities to destroy (at least 1)

    new_routes = []
    for r in routes:
        nr = [c for c in r if c not in removed]
        if nr:
            new_routes.append(nr) 
            # rebuild the route without those cities

    return new_routes, list(removed)


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
    for vn in removed:
        best_cost = float("inf")
        best_routes = None

        # try inserting vn into all existing routes
        for i in range(len(routes) + 1):
            # we test every possible insertion in the solution
            if i == len(routes):
                trial_routes = routes + [[vn]]
                # add the city as its own route as opposed to any other pre-existing route
                cost = routes_cost(trial_routes, problem, shortest)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = trial_routes
                continue

            for pos in range(len(routes[i]) + 1):
                trial_route = routes[i][:pos] + [vn] + routes[i][pos:]
                trial_routes = routes[:i] + [trial_route] + routes[i+1:]
     
                cost = routes_cost(trial_routes, problem, shortest)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = trial_routes

        if best_routes:
            routes = best_routes
        else:
            # fallback if somehow no insertion found (shouldn't happen with new route option)
            routes.append([vn])

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

