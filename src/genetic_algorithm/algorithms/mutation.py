from __future__ import annotations

import random

import numpy as np
from functools import reduce
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List, Optional, Protocol


class MutationAlgorithm(Protocol):
    """The bare type of a mutation algorithm"""

    @beartype
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        raise NotImplementedError


class BitflipMutation(MutationAlgorithm):
    @beartype
    def __init__(self, rate: float, lb: int = 0, ub: int = 1) -> None:
        """A uniform mutation algorithm

        ---
        Parameters:
        rate: float [0:1]
            The rate at which to randomly mutate any bit
        """
        self.rate = rate
        self.lb = lb
        self.ub = ub

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Uniform mutation algorithm on population

        ---
        Parameters:
        population: NDArray
            Array representing the population

        ---
        Returns:
        NDArray representing the offspring
        """
        # Numpy doesn't have map().
        # Instead, you define a function, vecotise it, and then apply it on the array.
        chance = self.rate / population.shape[1]
        flip_bits = np.vectorize(
            lambda x: x
            if (random.random() > chance)
            else np.random.randint(low=self.lb, high=self.ub + 1, dtype=np.int8)
        )
        return flip_bits(population)


class InsertionMutation(MutationAlgorithm):
    @beartype
    def __init__(self, rate: float = 1.0, multiple_values: bool = False) -> None:
        """An insertion mutation algorithm

        ---
        Parameters:
        rate: float [0:1]
            The rate at which to randomly mutate any individual
        multiple_values: bool
            If set to false, one value will be moved at a time. If true, groups are moved.
        """
        self.rate = rate
        self.multiple_values = multiple_values

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Insertion mutation algorithm on population

        ---
        Parameters:
        population: NDArray
            Array representing the population

        ---
        Returns:
        NDArray representing the offspring
        """
        for individual in population:
            if random.random() > self.rate:
                continue
            start, end = self._get_movement_boundries(dimensions=population.shape[1])
            shifts = self._get_number_of_shifts(start, end)
            individual[start:end] = np.roll(individual[start:end], shifts)

        return population

    @beartype
    @staticmethod
    def _get_movement_boundries(dimensions: int) -> List[np.int64]:
        """Gets the movement boundries: The two values from and to which things are moved

        ---
        Parameters:
        dimensions: int
            The amount of dimensions of the population & The range between the two numbers

        ---
        Returns:
        Array containing the two values
        """
        return sorted(np.random.choice(dimensions, size=2, replace=False))

    @beartype
    def _get_number_of_shifts(self, start: np.int64, end: np.int64) -> int:
        """Gets the amount of shifts to perform in between the two numbers

        ---
        Parameters:
        start: int
            The place the movement will start
        end: int
            The place the movement will end

        ---
        Returns
        int representing the amount of moves to do
        """
        if self.multiple_values:
            return random.randrange(end - start)
        else:
            return 1


class SwapMutation(MutationAlgorithm):
    @beartype
    def __init__(self, rate: float = 1.0) -> None:
        """A swap mutation algorithm

        ---
        Parameters:
        rate: float [0:1]
            The rate at which to randomly mutate any individual
        """
        self.rate = rate

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Swap mutation algorithm on population

        ---
        Parameters:
        population: NDArray
            Array representing the population

        ---
        Returns:
        NDArray representing the offspring
        """
        for individual in population:
            if random.random() > self.rate:
                continue
            first, second = self._get_swap_locations(dimensions=population.shape[1])
            individual[[first, second]] = individual[[second, first]]

        return population

    @beartype
    @staticmethod
    def _get_swap_locations(dimensions: int) -> NDArray:
        """Gets the swap locations: The two locations that are swapped with eachother

        ---
        Parameters:
        dimensions: int
            The amount of dimensions of the population & The range between the two numbers

        ---
        Returns:
        Array containing the two values
        """
        return np.random.choice(dimensions, size=2, replace=False)


class CombinedMutation(MutationAlgorithm):
    @beartype
    def __init__(self, *args: MutationAlgorithm) -> None:
        """A wrapper that allows for combination of mutation algorithms

        ---
        Parameters:
        args:
            The mutation algorithms to combine
        """
        # The Haskell is less abstruse:
        # \x -> foldr args x
        self.mutation_algorithm = lambda x: reduce(lambda result, f: f(result), args, x)

    @beartype
    def __call__(self, population: NDArray) -> NDArray:
        """Uniform mutation algorithm on population

        ---
        Parameters:
        population: NDArray
            Array representing the population

        ---
        Returns:
        NDArray representing the offspring
        """
        return self.mutation_algorithm(population)
