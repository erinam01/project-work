import random


def build_path_from_routes(routes, problem, shortest):
    """
    Convert routes into the final path format required by the assignment.
    """
    path = []
    for route in routes:
        for node in route:
            gold = problem.graph.nodes[node]["gold"]
            path.append((node, gold))
        path.append((0, 0))  # return to base after each route
    return path


def routes_cost(routes, problem, shortest):
    """
    Compute total cost of a solution represented as multiple routes.
    """
    total = 0.0

    for route in routes:
        current = 0
        load = 0.0

        for node in route:
            dist = shortest[current][node]
            total += dist + (problem.alpha * dist * load) ** problem.beta
            load += problem.graph.nodes[node]["gold"]
            current = node

        # return to base
        dist = shortest[current][0]
        total += dist + (problem.alpha * dist * load) ** problem.beta

    return total


# -------- NEIGHBOURHOOD OPERATORS --------

def move_city(routes):
    """
    Move a random city from one route to another.
    """
    routes = [r[:] for r in routes if r]

    if len(routes) < 2:
        return routes

    r1, r2 = random.sample(range(len(routes)), 2)
    city = random.choice(routes[r1])
    routes[r1].remove(city)
    routes[r2].append(city)

    if not routes[r1]:
        del routes[r1]

    return routes


def swap_cities(routes):
    """
    Swap two cities inside the same route.
    """
    routes = [r[:] for r in routes if len(r) > 1]
    if not routes:
        return routes

    r = random.choice(routes)
    i, j = random.sample(range(len(r)), 2)
    r[i], r[j] = r[j], r[i]
    return routes


def split_route(routes):
    """
    Split a long route into two.
    """
    routes = [r[:] for r in routes if len(r) > 2]
    if not routes:
        return routes

    r = random.choice(routes)
    cut = random.randint(1, len(r) - 1)
    routes.remove(r)
    routes.append(r[:cut])
    routes.append(r[cut:])
    return routes

def repair_routes(routes, nodes):
    """Ensure all cities appear at least once."""
    present = set()
    for r in routes:
        present.update(r)

    missing = [n for n in nodes if n not in present]

    if not routes:
        routes = []

    for n in missing:
        routes.append([n])

    return routes