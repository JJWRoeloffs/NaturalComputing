from __future__ import annotations

import csv

import numpy as np
from beartype import beartype
from beartype.typing import List, Dict

from cellular_automata import (
    AutomataObjectiveFunction,
    SimilarityMethod,
    CellularAutomata,
)


@beartype
def new_standard_problem(
    dimension: int,
    test: str = "OneMax",
    instance=1,
) -> IOHProblem:
    return ioh.get_problem(test, instance, dimension, "Integer")


@beartype
def _read_inputfile(inputfile: str) -> List[Dict]:
    with open(inputfile, encoding="utf-8-sig") as f:
        data = csv.DictReader(f)
        return list(data)


@beartype
def _parse_input(data: List[Dict]) -> List[Dict]:
    """Get a new dict where all the values are the old ones evaluated as python code."""
    return [{k: eval(v) for k, v in item.items()} for item in data]


@beartype
def get_input(inputfile: str) -> List[Dict]:
    return _parse_input(_read_inputfile(inputfile))


@beartype
def objective_function_from_input(
    item: Dict, similarity: SimilarityMethod
) -> AutomataObjectiveFunction:
    ca = CellularAutomata(
        rule=item["rule#"],
        k=item["k"],
        r=1,
    )
    return AutomataObjectiveFunction(
        ca=ca,
        similarity=similarity,
        ct=np.asarray(item["CT"]),
        t=item["T"],
        k=item["k"],
    )
