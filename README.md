# project-work
## Computational Intelligence Project – Optimization of Gold Collection Routes

This project addresses a graph-based optimization problem in which a thief must visit a set of cities to steal gold while minimizing travel cost.

Each city contains a certain amount of gold, and the cost of movement between cities depends on distances and problem parameters.

The objective is to compute a feasible path that respects problem constraints and returns the solution in the required format:

```python
[(c1, g1), (c2, g2), …, (cN, gN), (0, 0)]
```

where each pair represents a visited city and the gold collected, and (0, 0) denotes returning to the base.

# solution
The solution is based on metaheuristic optimization techniques:

1. Iterated Local Search (ILS)
ILS is used as the core solver. It iteratively improves routes through local modifications and perturbations. It performs well on small and medium-sized instances and provides fast convergence.

2. Large Neighborhood Search (LNS)
For larger instances (N ≥ 50), LNS is applied as a refinement step. It destroys part of the current solution and reconstructs it using greedy insertion, allowing the algorithm to escape local optima and improve global exploration.

The combination of ILS (intensification) and LNS (diversification) balances solution quality and computational time.

```
project-work/
│
├── s348365.py
├── Problem.py
└── src/
├── lns.py
├── solver.py
├── test.py
└── utils.py
```

# Requirements
Install dependencies with:
```
pip install -r base_requirements.txt
```

# How to Run
To run the experiments:
```
python -m src.test
```

This executes multiple configurations and compares the proposed methods with the baseline solution. To see the run time needed for each type of configuration, a `time_needed.csv` was provided to allow for estimation; this includes only the problem solution time and not the time for the Problem instance creation.
- **Small instances** (N < 100): All configurations are solved in under one second.
- **Large instances** (N = 100): The majority solve in under two minutes. However, outliers exist: five configurations exceed two minutes, and two exceed five minutes. These outliers share specific characteristics (Density = 0.2 and β ∈ {2.0,4.0}), where the combination of graph sparsity and high weight penalties increases computational complexity."

# Output
Finally, the system generates multiple CSV files that allow a detailed comparison of the performance of the different methods and clearly highlight the improvements achieved over the baseline solution. 
