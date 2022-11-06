from __future__ import annotations

import math
import random

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List, Callable, Iterator

from helpers import take_random_individual

class Swap:
    @beartype
    @staticmethod
    def random(part: NDArray, i: int = 0) -> NDArray:
        return np.random.permutation(part)

    @beartype
    @staticmethod
    def roll(part: NDArray, i: int = 0) -> NDArray:
        return part if i % 2 else np.roll(part, 1, axis=0)


class UniformCrossover:
    @beartype
    def __init__(
            self,
            *,
            amount_of_parents: int = 2,
            offspring_rate: float = 1.0,
            swap_function: Callable = Swap.roll
        ) -> None:
        """A uniform recombination algorithm
        
        ---
        Parameters:
        amount_of_parents: int
            The amount of parents per pool of recombination
        offspring_rate: float (>1)
            The ratio of offsprings per parent
        """
        self.amount_of_parents = amount_of_parents
        self.offspring_rate = offspring_rate
        self.swap = swap_function

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Uniform recombination algorithm on population
        
        ---
        Parameters:
        population: NDArray
            Array representing the population
            
        ---
        Returns:
        NDArray representing the offspring
        """
        amount_of_groups = math.ceil((population.shape[0]/self.amount_of_parents)*self.offspring_rate)
        
        if population.shape[0] > amount_of_groups*self.amount_of_parents:
            raise ValueError("Effective reproduction rate should be at least 1")

        children_groups = self._get_uniform_children(
                population = population,
                amount_of_groups = amount_of_groups,
                amount_per_group = self.amount_of_parents
            )
        return np.vstack(list(children_groups))

    @beartype
    def _get_uniform_children(
        self,
        *,
        population: NDArray,
        amount_of_groups: int,
        amount_per_group: int
    ) -> Iterator[NDArray]:
        for _ in range(amount_of_groups):
            parents = take_random_individual(population, amount=amount_per_group)
            chances = np.random.rand(parents.shape[1])
            yield self._get_uniform_child_group(parents=parents, chances=chances)
    
    @beartype
    def _get_uniform_child_group(self, *, parents: NDArray, chances: NDArray) -> NDArray:
        child_group = np.copy(parents)
        for i in range(parents.shape[1]): #Inefficient: There should be a way without looping over the dimensions in python.
            if chances[i] < 0.5:
                child_group[:, i] = self.swap(parents[:, i])
        return child_group


class PointCrossover:
    @beartype
    def __init__(
            self,
            *,
            amount_of_parents: int = 2,
            amount_of_splits: int = 1,
            offspring_rate: float = 1.0,
            swap_function: Callable = Swap.roll
        ) -> None:
        """A uniform recombination algorithm
        
        ---
        Parameters:
        amount_of_parents: int
            The amount of parents per pool of recombination
        amount_of_splits: int
            The amount of times the parents should be split to get recombined
        offspring_rate: float (>1)
            The ratio of offsprings per parent
        swap_function:
            The function to be used for swapping out (from the recombination.Swap class)
        """
        self.amount_of_parents = amount_of_parents
        self.offspring_rate = offspring_rate
        self.amount_of_splits = amount_of_splits
        self.swap = swap_function

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Point recombination algorithm on population
        
        ---
        Parameters:
        population: NDArray
            Array representing the population
            
        ---
        Returns:
        NDArray representing the offspring
        """
        amount_of_groups = math.ceil((population.shape[0]/self.amount_of_parents)*self.offspring_rate)

        if population.shape[0] > amount_of_groups*self.amount_of_parents:
            raise ValueError("Effective reproduction rate should be at least 1")

        children_groups = self._get_point_children(
                population = population,
                amount_of_groups = amount_of_groups,
                amount_per_group = self.amount_of_parents
            )
        return np.vstack(list(children_groups))


    @beartype
    def _get_point_children(
        self,
        *,
        population: NDArray,
        amount_of_groups: int,
        amount_per_group: int,
    ) -> Iterator[NDArray]:
        for _ in range(amount_of_groups):
            parents = take_random_individual(population, amount=amount_per_group)
            splits = self._get_random_ranges(dimensions = parents.shape[1])
            parts = list(self._get_split_parts(splits, parents))
            yield np.hstack(parts)

    @beartype
    def _get_random_ranges(self, *, dimensions: int) -> List[int]:
        """Get a list of integers representing the ranges for splitting an individual
        
        ---
        Parameters:
        dimensions: int
            amount of dimensions of the population to split
        self.amount_of_splits
            the amount of splits to make in every parent

        ---
        Returns:
        List of int representing the ranges to split over.
            Example: for one split at index 6 for 10 dimensions, the returned value would be [0,6,10]
            representing range [0:6] and [6:10]
        """
        return sorted(random.sample(range(1, dimensions), self.amount_of_splits) + [0, dimensions])

    @beartype
    def _get_split_parts(self, splits: List[int], parents: NDArray) -> Iterator[NDArray]:
        for i in range(len(splits)-1):
            part = parents[0:, splits[i]:splits[i+1]]
            yield self.swap(part, i)
