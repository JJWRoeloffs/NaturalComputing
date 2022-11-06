"""NACO assignment 22/23.

This file contains the skeleton code required to solve the first part of the 
assignment for the NACO course 2022. 

You can test your algorithm by using the function `test_algorithm`. For passing A1,
your GA should be able to pass the checks in this function for dimension=100.

## Installing requirements
You are encouraged to use a virtual environment. Install the required dependencies 
(via the command line) with the following command:
    pip install ioh>=0.3.3
"""
from __future__ import annotations

import random
import shutil
import sys
import math

import numpy as np
from beartype import beartype

import ioh

@beartype
def generate_rand_population(*, pop_size: int, dimensions: int) -> NDArray:
    return np.random.randint(2, size=(pop_size, dimensions), dtype = np.int8)



class Recombination:
    @beartype
    @classmethod
    def uniform(cls):
        pass

    @beartype
    @classmethod
    def point(cls):
        pass

class Mutation:
    @beartype
    @classmethod
    def bitflip(cls):
        pass

class Selection:
    @beartype
    @classmethod
    def tournament(cls, population):
        pass
    @beartype
    def roulette(cls, population):
        pass

class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

    def __init__(
            self,
            budget: int,
            pop_size: int = 20,
    ) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        budget: int
            The maximum number objective function evaluations
            the GA is allowed to do when solving a problem.

        Notes
        -----
        *   You can add more parameters to this constructor, which are specific to
            the GA or one of the (to be implemented) operators, such as a mutation rate.
        """

        self.budget = budget
        self.pop_size = pop_size

    def __call__(self, problem: ioh.problem.Integer) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.

        Notes
        -----
        *   This is the main body of you GA. You should implement all the logic
            for this search algorithm in this method. This does not mean that all
            the code needs to be in this method as one big block of code, you can
            use different methods you implement yourself.

        *   For now there is a random search process implemented here, which
            is a placeholder, and just to show you how to call the problem
            class.
        """

        population = generate_rand_population(
                pop_size=self.pop_size,
                dimensions=problem.meta_data.n_variables
            )

        while self.should_continue(problem):
            print(population)
            sys.exit(0)
            pass

        return problem.state.current_best

    def should_continue(self, problem: ioh.problem.Integer) -> bool:
        return problem.state.evaluations < self.budget and not problem.state.optimum_found


def test_algorithm(dimension, instance=1):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    instance: int
        The instance of the problem. Trying different instances of the problem,
        can be interesting if you want to check the robustness, of your GA.
    """

    budget = int(dimension * 5e3)
    problem = ioh.get_problem("OneMax", instance, dimension, "Integer")
    ga = GeneticAlgorithm(budget)
    solution = ga(problem)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert problem.state.evaluations >= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"OneMax was successfully solved in {dimension}D.\n")


def collect_data(dimension: int = 100, nreps: int = 5):
    """OneMax + LeadingOnes functions 10 instances.

    This function should be used to generate data, for A1.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    nreps: int
        The number of repetitions for each problem instance.
    """

    budget = int(dimension * 5e2)
    suite = ioh.suite.PBO([1, 2], list(range(1, 11)), [dimension])
    logger = ioh.logger.Analyzer(algorithm_name="GeneticAlgorithm")
    suite.attach_logger(logger)

    for problem in suite:
        print("Solving: ", problem)

        for _ in range(nreps):
            ga = GeneticAlgorithm(budget)
            ga(problem)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    # Simple test for development purpose
    test_algorithm(10)

    # Test required for A1, your GA should be able to pass this!
    # test_algorithm(100)

    # If your implementation passes test_algorithm(100)
    collect_data(100)
