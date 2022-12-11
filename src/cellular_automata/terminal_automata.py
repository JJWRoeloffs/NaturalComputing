from __future__ import annotations

import time
import argparse

from .automata import RuleSet
from .stage_renderer import StageRenderer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Print increasing stages of a cellular automata to the terminal
        Starting with a 1 in the center, printing ten new stages a second.
        """
    )
    parser.add_argument(
        "-w",
        "--wolfraam",
        nargs="?",
        default=30,
        type=int,
        help="The Wolfraam rule to use",
    )
    parser.add_argument(
        "-k",
        "--k",
        nargs="?",
        default=2,
        type=int,
        help="The amount of dimensions to use",
    )
    parser.add_argument(
        "-r",
        "--radius",
        nargs="?",
        default=1,
        type=int,
        help="The radius to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()

    stage = StageRenderer.get_simple_init()
    rule_set = RuleSet(args.wolfraam, args.k, args.radius)

    while True:
        StageRenderer.render(stage)
        stage = rule_set(stage)
        time.sleep(0.1)
