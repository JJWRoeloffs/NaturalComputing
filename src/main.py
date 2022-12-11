from beartype import beartype

from cellular_automata import AutomataObjectiveFunction
from genetic_algorithm.algorithms import *
from cellular_automata.similarity import *
from tests import get_input, objective_function_from_input, collect_data_cellular

INPUTFILE = "./input/ca_input.csv"


@beartype
def new_genetic_algorithm() -> GeneticAlgorithm:
    """Return a new genetic algorithem with the given amount of dimensions.
    Parameters of the algorithm can be set by writing code in this funcion"""

    crossover_algorithm = UniformCrossover(
        offspring_rate=1.7, amount_of_parents=4, swap_function=Swap.random
    )

    swap_algorithm = SwapMutation(rate=0.5)
    bitflip_algorithm = BitflipMutation(rate=0.5)
    mutation_algorithm = CombinedMutation(bitflip_algorithm, swap_algorithm)

    selection_algorithm = DeterministicSelection()

    return GeneticAlgorithm(
        pop_size=50,
        greedy=False,
        crossover_algorithm=crossover_algorithm,
        mutation_algorithm=mutation_algorithm,
        selection_algorithm=selection_algorithm,
    )


@beartype
def new_objective_function() -> AutomataObjectiveFunction:
    """Returns a new Automata Objective Function with the given data.
    Parameters can be set by writing code in this function"""

    similarity = EqualitySimilarity()
    input_data = get_input(INPUTFILE)
    return objective_function_from_input(input_data[1], similarity)


def main():
    collect_data_cellular(
        budget=10000,
        genetic_algorithm=new_genetic_algorithm(),
        objective_function=new_objective_function(),
        name="Hello There",
    )


if __name__ == "__main__":
    main()
