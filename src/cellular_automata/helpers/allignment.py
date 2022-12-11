from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype

# Written here because importing a library for it might not be "plain python."
# Algorithm is a litteral implementation of the Damerau paper's description.
@beartype
def damerau_levenshtein(x: NDArray, y: NDArray) -> int:
    matrix = np.zeros((len(x) + 1, len(y) + 1), dtype=np.int8)

    for i in range(len(x) + 1):
        matrix[i, 0] = i

    for j in range(len(y) + 1):
        matrix[0, j] = j

    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            cost = 0 if x[i-1] == y[j-1] else 1

            matrix[i, j] = min(
                [
                    matrix[i - 1, j] + 1,  # Deletion
                    matrix[i, j - 1] + 1,  # Insertion
                    matrix[i - 1, j - 1] + cost,  # Substitution
                ]
            )

            if (i > 1) and (j > 1) and (x[i-1] == y[j - 2]) and (x[i - 2] == y[j-1]):
                matrix[i, j] = min(
                    [
                        matrix[i, j],
                        matrix[i - 2, j - 2] + 1,  # Transportation
                    ]
                )

    return int(matrix[len(x), len(y)])
