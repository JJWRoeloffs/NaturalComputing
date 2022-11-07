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

import shutil

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import Callable

from helpers import generate_rand_population

import ioh

class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

    @beartype
    def __init__(
            self,
            pop_size: int,
            greedy: bool,
            recombination_algorithm: Callable,
            mutation_algorithm: Callable,
            selection_algorithm: Callable
    ) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        pop_size: int
            The population size to use.
        greedy: bool
            If the function should always include the current best in the new population
        recombination_algorithm: Callable
            The recombination algorithm to use
        mutation_algorithm: Callable
            The mutation algorithm to use
        selection-algorithm: Callable
            The selection algorithm to use
        """

        self.pop_size  = pop_size
        self.greedy    = greedy

        self.recombine = recombination_algorithm
        self.mutate    = mutation_algorithm
        self.select    = selection_algorithm

    @beartype
    def __call__(self, problem: ioh.problem.Integer, budget: int) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.
        budget: int
            The amount of times the GA is allowed to call the problem 
        """

        population = generate_rand_population(
                pop_size=self.pop_size,
                dimensions=problem.meta_data.n_variables
            )

        while self.should_continue(problem, budget):
            children = self.recombine(population)
            mutated_children = self.mutate(children)
            scores = self.evaluate(mutated_children, problem)
            population = self.select(children, scores, self.pop_size)
            if self.greedy:
                population = self.keep_current_best(population, problem)

        return problem.state.current_best

    @beartype
    @staticmethod
    def should_continue(problem: ioh.problem.Integer, budget: int) -> bool:
        """Whether the algorithm should continue one more generation or not

        ---
        Parameters:
        problem: ioh.problem.Integer
            The problem to evaluate
        budget: int
            The budget for the problem
        """
        return problem.state.evaluations < budget and not problem.state.optimum_found

    @beartype
    @staticmethod
    def evaluate(population: NDArray, problem: ioh.problem.Integer) -> NDArray:
        """Maps the problem on the population, returning a static list of scores

        ---
        Parameters:
        population: NDArray
            The population to evaluate
        problem: ioh.problem.Integer
            The problem to evaluate the population on

        ---
        Returns:
        NDArray of the scores of each individual, alligned by index
        """
        return np.asarray([problem(individual) for individual in population])
    
    @beartype
    @staticmethod
    def keep_current_best(population: NDArray, problem: ioh.problem.Integer) -> NDArray:
        """Appends the current best to the population for the next round
        
        ---
        Parameters:
        population: NDArray
            The population to append the current best to
        problem: ioh.problem.Integer
            The problem to find the current best from
        
        ---
        Returns:
        NDArray representing the new population
        """
        return np.vstack([[problem.state.current_best.x], population])

@beartype
def test_algorithm(genetic_algorithm: GeneticAlgorithm, dimension: int, type: str = "OneMax", instance=1):
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
    problem = ioh.get_problem(type, instance, dimension, "Integer")
    solution = genetic_algorithm(problem, budget)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert problem.state.evaluations <= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"OneMax was successfully solved in {dimension}D.\n")

@beartype
def collect_data(genetic_algorithm: GeneticAlgorithm, dimension: int, nreps: int = 5):
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
            genetic_algorithm(problem, budget)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")

@beartype
def new_genetic_algorithm() -> GeneticAlgorithm:
    """Return a new genetic algorithem with the given amount of dimensions.
    Parameters of the algorithm can be set by writing code in this funcion"""

    from algorithms import UniformCrossover, BitflipMutation, TournamentSelection, Swap

    recombination_algorithm = UniformCrossover(
        offspring_rate=1.7,
        amount_of_parents=4,
        swap_function = Swap.random
    )

    mutation_algorithm = BitflipMutation(
        rate = 0.01
    )

    selection_algorithm = TournamentSelection(
        remove_chosen=True
    )

    return GeneticAlgorithm(
        pop_size = 5,
        greedy   = True,
        recombination_algorithm = recombination_algorithm,
        mutation_algorithm = mutation_algorithm,
        selection_algorithm = selection_algorithm
    )

if __name__ == "__main__":
    # Simple test for development purpose
    # test_algorithm(new_genetic_algorithm(), 10)

    # Large tests
    # test_algorithm(new_genetic_algorithm(), 100)
    # test_algorithm(new_genetic_algorithm(), 100, type="LeadingOnes")

    # If your implementation passes test_algorithm(new_genetic_algorithm(), 100)
    collect_data(new_genetic_algorithm(), 100)
