from src.solver import solve


class Solver:
    def __init__(self, problem):
        self.problem = problem
        self._solution = None

    def solution(self, use_lns=False):
        if self._solution is None:
            self._solution = solve(self.problem, use_lns = use_lns)
        return self._solution
