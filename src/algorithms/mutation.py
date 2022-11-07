from __future__ import annotations

import random

import numpy as np
from nptyping import NDArray
from beartype import beartype

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
