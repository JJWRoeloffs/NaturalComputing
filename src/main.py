from __future__ import annotations

from beartype import beartype

from algorithms import *
from tests import test_algorithm, collect_data

@beartype
def new_genetic_algorithm() -> GeneticAlgorithm:
    """Return a new genetic algorithem with the given amount of dimensions.
    Parameters of the algorithm can be set by writing code in this funcion"""

    crossover_algorithm = UniformCrossover(
        offspring_rate=1.7,
        amount_of_parents=4,
        swap_function = Swap.random
    )

    mutation_algorithm = BitflipMutation(
        rate = 1.0
    )

    selection_algorithm = TournamentSelection(
        remove_chosen=True
    )

    return GeneticAlgorithm(
        pop_size = 5,
        greedy   = True,
        crossover_algorithm = crossover_algorithm,
        mutation_algorithm = mutation_algorithm,
        selection_algorithm = selection_algorithm
    )

if __name__ == "__main__":
    # Simple test for development purpose
    # test_algorithm(new_genetic_algorithm(), 10)

    # Large tests
    # test_algorithm(new_genetic_algorithm(), 100)
    # test_algorithm(new_genetic_algorithm(), 100, test="LeadingOnes")

    # If your implementation passes test_algorithm(new_genetic_algorithm(), 100)
    collect_data(new_genetic_algorithm(), 100)
