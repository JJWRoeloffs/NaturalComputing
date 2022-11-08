from __future__ import annotations

import math
import random

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List, Callable, Iterator, Protocol

from helpers import take_random_individual

class CrossoverAlgorithm(Protocol):
    """The bare type of a crossover algorithm"""
    @beartype
    def __init__(*args, **kwargs) -> None:
        raise NotImplementedError

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        raise NotImplementedError

class Swap:
    @beartype
    @staticmethod
    def random(part: NDArray, i: int = 0) -> NDArray:
        """Random Swap funcion
        This function can be used in the crossover algorithms to swap parts' places
        It returns the parts in random order

        ---
        Parameters:
        part: NDArray
            Numpy array representing the part
        i: int
            Index at which the part is (not used by this function)

        ---
        Returns:
        NDArray representing the new swapped parts.
        """
        return np.random.permutation(part)

    @beartype
    @staticmethod
    def roll(part: NDArray, i: int = 0) -> NDArray:
        """Rolling Swap funcion
        This function can be used in the crossover algorithms to swap parts' places
        It returns the parts in rolled order, based on the provided index
            Example: at i=1 [1,2,3,4] would roll to [4,1,2,3]

        ---
        Parameters:
        part: NDArray
            Numpy array representing the part
        i: int
            Index at which the part is

        ---
        Returns:
        NDArray representing the new swapped parts.
        """
        return np.roll(part, i, axis=0)


class UniformCrossover(CrossoverAlgorithm):
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

        if self.amount_of_parents > population.shape[0]:
            raise ValueError("Amount of parents must be smaller than population size")

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
        """Get uniformly-crossed-over children from population

        ---
        Parameters:
        population: NDArray
            Numpy array representing the population to make children from
        amount_of_groups: int
            The amount of child groups to make
        amount_per_group: int
            The amount of individuals (parents & children) per crossed-over group

        ---
        Returns:
        Itterator over the generated child groups
        """
        for _ in range(amount_of_groups):
            parents = take_random_individual(population, amount=amount_per_group)
            chances = np.random.rand(parents.shape[1])
            yield self._get_uniform_child_group(parents=parents, chances=chances)

    @beartype
    def _get_uniform_child_group(self, *, parents: NDArray, chances: NDArray) -> NDArray:
        """Get Uniformly-crossed-over child group from population
        Each part will be swapped/randomised according to self.swap
        NOTE: This algorithm is slow, and can do with improving

        ---
        Parameters:
        chances: NDArray
            Numpy array representing the changes for each index to swap over
        parents: NDArray
            Numpy array representing the parent group to return the parts from
        self.swap: Callable
            The function to swap the parts with (From the Swap class)

        ---
        Returns:
        NDArray representing the new child group
        """
        child_group = np.copy(parents)
        for i in range(parents.shape[1]): #Inefficient: There should be a way without looping over the dimensions in python.
            if chances[i] < 0.5:
                child_group[:, i] = self.swap(parents[:, i])
        return child_group


class PointCrossover(CrossoverAlgorithm):
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

        if self.amount_of_parents > population.shape[0]:
            raise ValueError("Amount of parents must be smaller than population size")

        if self.amount_of_splits > population.shape[1]:
            raise ValueError("Amount of splits has to be smaller than the dimensions of the individuals")

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
        """Get point-crossed-over children from population

        ---
        Parameters:
        population: NDArray
            Numpy array representing the population to make children from
        amount_of_groups: int
            The amount of child groups to make
        amount_per_group: int
            The amount of individuals (parents & children) per crossed-over group

        ---
        Returns:
        Itterator over the generated child groups
        """
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
        """Get Itterator over parts, representing the individual sections from a group of parents
        Each part will be swapped/randomised according to self.swap

        ---
        Parameters:
        splits: List[int]
            List representing the splits that need to be done (formatted as ._get_random_ranges)
        parents: NDArray
            Numpy array representing the parent group to return the parts from
        self.swap: Callable
            The function to swap the parts with (From the Swap class)

        ---
        Returns:
        Itterator over the generated parts
        """
        for i in range(len(splits)-1):
            part = parents[0:, splits[i]:splits[i+1]]
            yield self.swap(part, i)
