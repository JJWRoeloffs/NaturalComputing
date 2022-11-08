from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype

from helpers import generate_rand_population
from .crossover import CrossoverAlgorithm
from .mutation import MutationAlgorithm
from .selection import SelectionAlgorithm

import ioh

class GeneticAlgorithm:
    @beartype
    def __init__(
            self,
            pop_size: int,
            greedy: bool,
            crossover_algorithm: CrossoverAlgorithm,
            mutation_algorithm: MutationAlgorithm,
            selection_algorithm: SelectionAlgorithm
    ) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        pop_size: int
            The population size to use.
        greedy: bool
            If the function should always include the current best in the new population
        crossover_algorithm: Callable
            The crossover algorithm to use
        mutation_algorithm: Callable
            The mutation algorithm to use
        selection-algorithm: Callable
            The selection algorithm to use
        """

        self.pop_size  = pop_size
        self.greedy    = greedy

        self.crossover = crossover_algorithm
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
            children = self.crossover(population)
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
