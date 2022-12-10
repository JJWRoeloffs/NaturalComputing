from __future__ import annotations

import os

import numpy as np
from nptyping import NDArray
from beartype import beartype


class StageRenderer:
    characters = [
        " ",
        "█",
        "▒",
        "░",
        "▓",
    ]

    @beartype
    @classmethod
    def render_item(cls, item: np.int8) -> str:
        """Returns a shaded unicode full block for an item"""
        if item > 4:
            raise NotImplementedError

        # The numbers are nominals
        # mapping to increasing opaqueness is not needd.
        return cls.characters[item]

    @beartype
    @classmethod
    def render(cls, stage: NDArray):
        """render the given stage with this rule"""
        print("".join(map(cls.render_item, stage)))

    @beartype
    @staticmethod
    def get_simple_init() -> NDArray:
        """Returns an array of the width of the screen that's all 0s with one 1 in the middle.
        Usefull to get a starting point when printing
        """
        screen_width = int(os.popen("stty size", "r").read().split()[1])
        init = np.zeros(screen_width, dtype=np.int8)
        init[int(screen_width / 2)] = 1
        return init
