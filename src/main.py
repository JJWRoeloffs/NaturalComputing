from beartype import beartype
from beartype.typing import Dict

from cellular_automata import AutomataObjectiveFunction
from genetic_algorithm.algorithms import *
from cellular_automata.similarity import *
from tests import *

INPUTFILE = "./input/ca_input.csv"


@beartype
def new_genetic_algorithm(
    objective_function: AutomataObjectiveFunction,
) -> GeneticAlgorithm:
    """Return a new genetic algorithem with the given amount of dimensions.
    Parameters of the algorithm can be set by writing code in this funcion"""

    crossover_algorithm = UniformCrossover(
        offspring_rate=1.7, amount_of_parents=4, swap_function=Swap.random
    )

    swap_algorithm = SwapMutation(rate=0.5)
    bitflip_algorithm = BitflipMutation(rate=0.5, ub=objective_function.k - 1)
    mutation_algorithm = CombinedMutation(bitflip_algorithm, swap_algorithm)

    selection_algorithm = DeterministicSelection()

    return GeneticAlgorithm(
        pop_size=50,
        greedy=True,
        crossover_algorithm=crossover_algorithm,
        mutation_algorithm=mutation_algorithm,
        selection_algorithm=selection_algorithm,
        objective_function=objective_function,
    )


@beartype
def new_objective_function(input_data: Dict) -> AutomataObjectiveFunction:
    """Returns a new Automata Objective Function with the given data.
    Parameters can be set by writing code in this function"""

    similarity = ValueRespectingSimilarity(k=input_data["k"])
    return objective_function_from_input(input_data, similarity)


def main():
    input_data = get_input(INPUTFILE)
    objective_function = new_objective_function(input_data[8])
    genetic_algorithm = new_genetic_algorithm(objective_function)
    problem = wrap_objective_function(objective_function, "Test")

    test_algorithm(
        budget=10000,
        genetic_algorithm=genetic_algorithm,
        problem=problem,
    )

    collect_data_cellular(
        budget=10000,
        genetic_algorithm=genetic_algorithm,
        problem=problem,
    )


if __name__ == "__main__":
    main()
