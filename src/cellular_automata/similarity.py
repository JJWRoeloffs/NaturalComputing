from __future__ import annotations

from nptyping import NDArray
from beartype import beartype
from beartype.typing import Protocol


class SimilarityMethod(Protocol):
    """The Bare type of a Similarity Method"""

    @beartype
    def __init__(*args, **kwargs) -> None:
        raise NotImplementedError

    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        raise NotImplementedError


class EqualitySimilarity(SimilarityMethod):
    @beartype
    def __init__(self) -> None:
        pass

    @beartype
    @staticmethod
    def __call__(ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns the amount of overlapping inputs
        ---
        Parameters:
        ct: NDArray
            The perfect output
        ct_prime: NDArray
            The suggested output
        ---
        Returns:
        float
            The Similarity
        """
        return float(sum([int(x == y) for x, y in zip(ct, ct_prime)]))


class ValueRespectingSimilarity(SimilarityMethod):
    @beartype
    def __init__(self, dimensions: int):
        self.similarity = lambda x, y: dimensions - ((x - y) % dimensions)

    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns the amount of overlapping inputs
        ---
        Parameters:
        ct: NDArray
            The perfect output
        ct_prime: NDArray
            The suggested output
        ---
        Returns:
        float
            The Similarity
        """
        return float(sum([similarity(x, y) for x, y in zip(ct, ct_prime)]))
