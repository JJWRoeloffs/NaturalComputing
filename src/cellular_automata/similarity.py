from __future__ import annotations

from difflib import SequenceMatcher

from nptyping import NDArray
from beartype import beartype
from beartype.typing import Protocol

from .helpers import damerau_levenshtein, damerau_levenshtein_imported


class SimilarityMethod(Protocol):
    """The Bare type of a Similarity Method"""

    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        raise NotImplementedError


class HammingSimilarity(SimilarityMethod):
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
        return float(sum([int(x == y) for x, y in zip(ct, ct_prime)]))


class LeeSimilarity(SimilarityMethod):
    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns the amount of overlapping inputs,
        whilst respecting the integer differences
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
        return float(sum([abs(x - y) for x, y in zip(ct, ct_prime)]))


class DamerauLevenshteinSimilarity(SimilarityMethod):
    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns Damerau Levenshtein Distance
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
        return float(len(ct) - damerau_levenshtein_imported(ct, ct_prime))


class LCSSimilarity(SimilarityMethod):
    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns the length of the longest common subsequence
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
        return float(SequenceMatcher(None, ct, ct_prime).find_longest_match().size)


class GestaltSimilarity(SimilarityMethod):
    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        """A similarity function that returns the Gestalt Similarity
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
        return float(2* len(ct) * SequenceMatcher(None, ct, ct_prime).ratio())
