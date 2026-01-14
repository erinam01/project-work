import random
from .utils import routes_cost, repair_routes


# --------------------------------------------------
# DESTROY OPERATOR
# --------------------------------------------------

def destroy(routes, fraction=0.25):
    """
    Remove a fraction of cities from the current solution.
    Returns:
      - partial routes
      - list of removed cities
    """
    all_cities = [c for r in routes for c in r]
    if not all_cities:
        return routes, []

    k = max(1, int(len(all_cities) * fraction))
    removed = set(random.sample(all_cities, k))

    new_routes = []
    for r in routes:
        nr = [c for c in r if c not in removed]
        if nr:
            new_routes.append(nr)

    return new_routes, list(removed)


# --------------------------------------------------
# GREEDY REPAIR OPERATOR
# --------------------------------------------------

def greedy_repair(routes, removed, problem, shortest):
    """
    Reinsert removed cities in the cheapest position
    (best marginal cost insertion).
    """
    nodes = [n for n in problem.graph.nodes if n != 0]

    for city in removed:
        best_cost = float("inf")
        best_routes = None

        # Try inserting city in every possible position
        for i in range(len(routes) + 1):
            if i == len(routes):
                trial_routes = routes + [[city]]
            else:
                for pos in range(len(routes[i]) + 1):
                    trial = routes[i][:pos] + [city] + routes[i][pos:]
                    trial_routes = routes[:i] + [trial] + routes[i + 1 :]

                    trial_routes = repair_routes(trial_routes, nodes)
                    cost = routes_cost(trial_routes, problem, shortest)

                    if cost < best_cost:
                        best_cost = cost
                        best_routes = trial_routes

        if best_routes is not None:
            routes = best_routes
        else:
            routes.append([city])

    return routes


# --------------------------------------------------
# LNS MAIN LOOP
# --------------------------------------------------

def lns(problem, shortest,
        start_routes,
        iterations=100,
        destroy_fraction=0.25):
    """
    Large Neighborhood Search starting from a given solution.
    """
    nodes = [n for n in problem.graph.nodes if n != 0]

    best_routes = start_routes
    best_cost = routes_cost(best_routes, problem, shortest)

    for _ in range(iterations):
        # Destroy
        partial, removed = destroy(best_routes, destroy_fraction)

        # Repair
        candidate = greedy_repair(partial, removed, problem, shortest)
        candidate = repair_routes(candidate, nodes)

        cost = routes_cost(candidate, problem, shortest)

        if cost < best_cost:
            best_routes = candidate
            best_cost = cost

    return best_routes
