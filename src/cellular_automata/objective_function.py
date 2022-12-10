from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List

from .automata import CellularAutomata
from .similarity import SimilarityMethod


class AutomataObjectiveFunction:
    @beartype
    def __init__(
        self, ca: CellularAutomata, similarity: SimilarityMethod, ct: NDArray, t: int
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

    @beartype
    def __call__(self, c0_prime: NDArray) -> float:
        """
        ---
        Parameters
        c0_prime: NDArray
            A suggested c0 state

        Returns
        -------
        float
            The similarity of ct_prime to the true ct state of the CA
        """
        ct_prime = self.ca(c0_prime, self.t)
        return similarity(self.ct, ct_prime)
