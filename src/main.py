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

    crossover_algorithm = PointCrossover(
        offspring_rate=1.7, amount_of_parents=4, swap_function=Swap.random, amount_of_splits=4
    )

    # crossover_algorithm = UniformCrossover(
    #         offspring_rate = 1.8, amount_of_parents=4, swap_function=Swap.random
    #     )

    insertion_algorithm = InsertionMutation(rate=0.5, multiple_values=True)
    bitflip_algorithm = BitflipMutation(
        rate=0.5, ub=objective_function.ca.rule_set.k - 1
    )
    mutation_algorithm = CombinedMutation(bitflip_algorithm, insertion_algorithm)

    selection_algorithm = TournamentSelection()

    return GeneticAlgorithm(
        pop_size=100,
        greedy=False,
        crossover_algorithm=crossover_algorithm,
        mutation_algorithm=mutation_algorithm,
        selection_algorithm=selection_algorithm,
        objective_function=objective_function,
    )


@beartype
def new_objective_function(input_data: Dict) -> AutomataObjectiveFunction:
    """Returns a new Automata Objective Function with the given data.
    Parameters can be set by writing code in this function"""

    similarity = HammingSimilarity()
    return objective_function_from_input(input_data, similarity)


def main():
    input_data = get_input(INPUTFILE)
    objective_function = new_objective_function(input_data[0])
    genetic_algorithm = new_genetic_algorithm(objective_function)
    problem = wrap_objective_function(objective_function, "Test")

    # test_algorithm(
    #     budget=10000,
    #     genetic_algorithm=genetic_algorithm,
    #     problem=problem,
    # )

    collect_data_cellular(
        budget=10000,
        genetic_algorithm=genetic_algorithm,
        problem=problem,
        name="HammingInsertion1"
    )


if __name__ == "__main__":
    main()
