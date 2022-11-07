from __future__ import annotations

import random

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List

class BitflipMutation:
    @beartype
    def __init__(self, rate: float) -> None:
        """A uniform mutation algorithm

        ---
        Parameters:
        rate: float [0:1]
            The rate at which to randomly mutate any bit
        """
        # Numpy doesn't have map().
        # Instead, you define a function, vecotise it, and then apply it on the array.
        self.flip_bits = np.vectorize(lambda x: x if (random.random() > rate) else int(not x))

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
        return self.flip_bits(population)

class InsertionMutation:
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
        """Uniform mutation algorithm on population

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
            return random.randrange(end-start)
        else:
            return 1

class CombinedMutation:
    @beartype
    def __init__(self, bitflip_algorithm: BitflipMutation, insertion_algorithm: InsertionMutation) -> None:
        """A wrapper that combines Bitflip and Insertion mutation

        ---
        Parameters:
        bitflip_algorithm: BitflipMutation
            The bitflip algorithm to be used
        insertion_algorithm: InsertionMutation
            The Insertion algorithm to be used
        """
        self.bitflip_algorithm = bitflip_algorithm
        self.insertion_algorithm = insertion_algorithm

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
        population = self.insertion_algorithm(population)
        return self.bitflip_algorithm(population)


