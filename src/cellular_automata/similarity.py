from __future__ import annotations

from nptyping import NDArray
from beartype import beartype
from beartype.typing import Protocol

class SimilatityMethod(Protocol):
    """The Bare type of a Similarity Method"""

    @beartype
    def __init__(*args, **kwargs) -> None:
        raise NotImplementedError

    @beartype
    def __call__(self, ct: NDArray, ct_prime: NDArray) -> float:
        raise NotImplementedError
