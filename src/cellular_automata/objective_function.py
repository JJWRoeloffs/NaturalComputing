import beartype
from beartype.typing import List

from .automata import CellularAutomata

def make_objective_function(ct, rule, t, similarity_method):
    """Create a CA objective function."""

    if similarity_method == 1:

        def similarity(ct: List[int], ct_prime: List[int]) -> float:
            """You should implement this"""

            return random.uniform(0, 100)

    else:

        def similarity(ct: List[int], ct_prime: List[int]) -> float:
            """You should implement this"""

            return random.normalvariate(0, 10)

    def objective_function(c0_prime: List[int]) -> float:
        """Skeleton objective function.

        You should implement a method  which computes a similarity measure
        between c0_prime a suggested by your GA, with the true c0 state
        for the ct state given in the sup. material.

        Parameters
        ----------
        c0_prime: list[int] | np.ndarray
            A suggested c0 state

        Returns
        -------
        float
            The similarity of ct_prime to the true ct state of the CA
        """

        ca = CellularAutomata(rule)
        ct_prime = ca(c0_prime, t)
        return similarity(ct, ct_prime)

    return objective_function
