from Problem import Problem
from src.solver import solve


def solution(p:Problem):
        if getattr(p, "_solution", None) is None:
            p._solution = solve(p)
        return p._solution