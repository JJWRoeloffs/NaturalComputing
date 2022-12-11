from __future__ import annotations

import shutil

from beartype import beartype
from beartype.typing import Union, Callable

from .helpers import new_standard_problem, wrap_objective_function
from genetic_algorithm.algorithms import GeneticAlgorithm
from cellular_automata import (
    AutomataObjectiveFunction,
    SimilarityMethod,
    CellularAutomata,
)

import ioh


@beartype
def test_algorithm(
    budget: int,
    genetic_algorithm: GeneticAlgorithm,
    problem: Union[str, Callable] = "OneMax",
    dimension: int = 100,
):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.
    problem: str
        The type of problem to test on (OneMax or LeadingOnes)
    """

    if isinstance(problem, str):
        problem = new_standard_problem(problem, dimension)
    solution = genetic_algorithm(problem, budget)

    print("GA found solution:\n", solution)

    assert problem.state.evaluations <= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"{test} was successfully solved in {dimension}D.\n")


@beartype
def collect_data_onemax_leadingones(
    genetic_algorithm: GeneticAlgorithm,
    name: str,
    dimension: int,
    nreps: int = 5,
):
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


def collect_data_cellular(
    budget: int,
    genetic_algorithm: GeneticAlgorithm,
    problem: Callable,
    nreps: int = 9,
):
    """CellularAutomata evaluation

    This function should be used to generate data, for A2.

    Parameters
    ----------
    budget: int
        The budget
    genetic_algorithm: GeneticAlgorithm
        The algorithm to test
    problem: Callable
        The ioh problem function to use
    name: str
        The name to use for the algorithm
    nreps: int
        The number of repetitions for each problem instance.
    """

    logger = ioh.logger.Analyzer()
    problem.attach_logger(logger)

    for _ in range(nreps):
        genetic_algorithm(problem, budget)
        problem.reset()

    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")
