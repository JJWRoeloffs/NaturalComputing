from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype

@beartype
def generate_rand_population(*, pop_size: int, dimensions: int) -> NDArray:
    """Generates a random population

    ---
    Parameters:
    pop_size: int
        Population size of the population
    dimensions: int
        amount of dimensions for each individual in the population

    ---
    Returns:
    NDArray[NDarray[np.int8]]
        A two-dimensional array with 8-bit ints of random values 1 or 0  
    """
    return np.random.randint(2, size=(pop_size, dimensions), dtype = np.int8)

@beartype
def take_random_individual(population: NDArray, *, amount: int = 1) -> NDArray:
    """Takes random distinct individuals from a population

    ---
    Parameters:
    population: NDArray[NDarray[np.int8]]
        A two-dimensional array representing the population to pull from
    amount: int
        The amount of non-repeating individuals to take

    ---
    Returns:
    NDArray[NDArray[np.int8]]
        A two-dimensional array representing the picked individuals.
    """
    return population[np.random.choice(len(population), size=amount, replace=False)]
