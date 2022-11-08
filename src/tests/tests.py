from __future__ import annotations

import shutil

from beartype import beartype

from algorithms import GeneticAlgorithm

import ioh

@beartype
def test_algorithm(genetic_algorithm: GeneticAlgorithm, dimension: int, test: str = "OneMax", instance=1):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.
    test: str
        The type of problem to test on (OneMax or LeadingOnes)
    instance: int
        The instance of the problem. Trying different instances of the problem,
        can be interesting if you want to check the robustness, of your GA.
    """

    budget = int(dimension * 5e3)
    problem = ioh.get_problem(test, instance, dimension, "Integer")
    solution = genetic_algorithm(problem, budget)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert problem.state.evaluations <= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"{test} was successfully solved in {dimension}D.\n")

@beartype
def collect_data(genetic_algorithm: GeneticAlgorithm, name: str, dimension: int, nreps: int = 5):
    """OneMax + LeadingOnes functions 10 instances.

    This function should be used to generate data, for A1.

    Parameters
    ----------
    genetic_algorithm: GeneticAlgorithm
        The algorithm to test
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.
    name: str
        The name to use for the algorithm
    nreps: int
        The number of repetitions for each problem instance.
    """

    budget = int(dimension * 5e2)
    suite = ioh.suite.PBO([1, 2], list(range(1, 11)), [dimension])
    logger = ioh.logger.Analyzer(algorithm_name=name)
    suite.attach_logger(logger)

    for problem in suite:
        print("Solving: ", problem)

        for _ in range(nreps):
            genetic_algorithm(problem, budget)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")
