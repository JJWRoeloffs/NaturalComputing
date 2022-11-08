from __future__ import annotations

import random
import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import Protocol

class SelectionAlgorithm(Protocol):
    """The bare type of a selection algorithm"""
    @beartype
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @beartype
    def __call__(self, children: NDArray, scores: NDArray, result_size: int) -> NDArray:
        raise NotImplementedError

class TournamentSelection(SelectionAlgorithm):
    @beartype
    def __init__(self, remove_chosen: bool = False, amount_to_take: int = 2) -> None:
        """A tournament selection algorithm

        ---
        Parameters:
        remove_chosen: bool
            Whether or not to remove previously chosen individuals from the tournament pool
        amount_to_take: int
            The amount of children to take at once for each tournament round
        """
        self.remove_chosen = remove_chosen
        self.amount_to_take = amount_to_take

    @beartype
    def __call__(self, children: NDArray, scores: NDArray, result_size: int) -> NDArray:
        """Tournament selection algorithm

        ---
        Parameters:
        children: NDArray
            Array representing the children to select from
        scores: List[int]
            Array representing the scores of each individual in the population
            Each index should represent the same index in the population
        result_size: int
            The population size of the result

        ---
        Returns:
        NDArray representing the new population
        """
        output = []
        while len(output) < result_size:
            winner_index = self.tournament_round(scores)
            output.append(children[winner_index])
            if self.remove_chosen:
                children = np.delete(children, winner_index, axis = 0)
                scores   = np.delete(scores, winner_index, axis = 0)
        return np.asarray(output)

    @beartype
    def tournament_round(self, scores: NDArray) -> np.int64:
        """Get the results of one round of the tournament

        ---
        Parameters:
        scores: NDArray
            The Numpy array of scores to chose from

        ---
        Returns:
        np.int46, the index of the winning individual
        """
        selection = scores[
                (indexes := np.random.choice(len(scores), size=self.amount_to_take, replace=False))
            ]
        return indexes[selection.argmax()]

class RouletteSelection(SelectionAlgorithm):
    @beartype
    def __init__(self, remove_chosen: bool = False) -> None:
        """A roulette selection algorithm

        ---
        Parameters:
        rate: float [0:1]
            The rate at which to randomly mutate 
        """
        self.remove_chosen = remove_chosen

    @beartype
    def __call__(self, children: NDArray, scores: NDArray, result_size: int) -> NDArray:
        """Roulette selection algorithm

        ---
        Parameters:
        children: NDArray
            Array representing the children to select from
        scores: List[int]
            Array representing the scores of each individual in the population
            Each index should represent the same index in the population
        result_size: int
            The population size of the result

        ---
        Returns:
        NDArray representing the new population
        """
        output = []
        while len(output) < result_size:
            winner_index = self.roulette_wheel(scores)
            output.append(children[winner_index])
            if self.remove_chosen:
                children = np.delete(children, winner_index, axis = 0)
                scores   = np.delete(scores, winner_index, axis = 0)
        return np.asarray(output)

    @beartype
    def roulette_wheel(self, scores: NDArray) -> np.int64:
        """Gets the result of a single roulette spin

        ---
        Parameters:
        scores: NDArray
            The Numpy array of scores to chose from

        ---
        Returns:
        np.int46, the index of the winning individual
        """
        [value] = random.choices(scores, scores)
        return np.where(scores==value)[0][0]
