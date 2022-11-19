import shutil
import random

import ioh

from genetic_algorithm import GeneticAlgorithm
from .cellular_automata import make_objective_function


def example(nreps=10):
    """An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer."""

    ct, rule, t = None, None, None  # Given by the sup. material

    # Create an objective function
    objective_function = make_objective_function(ct, rule, t, 1)

    # Wrap objective_function as an ioh problem
    problem = ioh.wrap_problem(
        objective_function,
        name="objective_function_ca_1",  # Give an informative name
        dimension=10,  # Should be the size of ct
        problem_type="Integer",
        optimization_type=ioh.OptimizationType.MAX,
        lb=0,
        ub=1,  # 1 for 2d, 2 for 3d
    )
    # Attach a logger to the problem
    logger = ioh.logger.Analyzer()
    problem.attach_logger(logger)

    # run your algoritm on the problem
    for _ in range(nreps):
        algorithm = GeneticAlgorithm(10)
        algorithm(problem)
        problem.reset()

    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    example()
