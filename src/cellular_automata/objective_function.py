from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List, Callable

from .automata import CellularAutomata
from .similarity import SimilarityMethod, EqualitySimilarity


class AutomataObjectiveFunction:
    @beartype
    def __init__(
        self,
        ca: CellularAutomata,
        similarity: SimilarityMethod,
        ct: NDArray,
        t: int,
        k: int,
    ) -> None:
        """Automata objective function: calculate the quality if the input

        ---
        Parameters:
        ca: CellularAutomata
            The Cellular Automata to use for the evaulation
        similarity: SimilarityMethod
            The similarity method to use for the evaulation
        ct: NDArray
            the expected result
        t: int
            the amount of steps to take to get to the expected result
        """
        self.similarity = similarity
        self.ca = ca
        self.ct = ct
        self.t = t
        self.k = k

    @beartype
    def get_function(self) -> Callable:
        """
        Return a "clean" objective function that can be properly mapped to a C++ type.

        Returns
        -------
        Callable
            The objective function to use.
        """

        def objective_function(c0_prime: NDArray) -> float:
            ct_prime = self.ca(c0_prime, self.t)
            return self.similarity(self.ct, ct_prime)

        return objective_function

    @beartype
    def is_optimal(self, c0_prime: NDArray) -> bool:
        """Is the current best optimal?
        ---
        Parameters:
        c0_prime: NDArray
            The current best to test
        ---
        Returns
        bool"""
        return len(c0_prime) == EqualitySimilarity()(self.ct, c0_prime)
