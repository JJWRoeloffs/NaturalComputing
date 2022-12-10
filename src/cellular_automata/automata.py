from __future__ import annotations

import numpy as np
from nptyping import NDArray
from beartype import beartype
from beartype.typing import List, Dict


class CellularAutomata:
    @beartype
    def __init__(self, rule: int, k: int = 2, r: int = 1):
        """A Cellular Atuomaton"""
        self.rule_set = RuleSet(rule, k, r)

    @beartype
    def __call__(self, stage: List[np.int8], t: int) -> List[np.int8]:
        """Evaluate for T timesteps. Return Ct for a given C0."""
        for _ in range(t):
            stage = rule_set(stage)
        return stage


class RuleSet:
    @beartype
    def __init__(self, rule: int, k: int = 2, r: int = 1):
        self.rule = rule
        self.k = k
        self.r = r
        self.rule_dict = self.get_ruleset_dict(rule, k, r)

    @beartype
    @staticmethod
    def get_ruleset(rule: int, k: int = 2, r: int = 1) -> List[np.int8]:
        length = k ** (2 * r + 1)
        rulestring = np.base_repr(rule, k).zfill(length)
        return list(reversed([np.int8(x) for x in rulestring]))

    @beartype
    @classmethod
    def get_ruleset_dict(cls, rule: int, k: int = 2, r: int = 1) -> Dict[str, np.int8]:
        ruleset = cls.get_ruleset(rule, k, r)

        rule_dict = {}
        for i, result in enumerate(ruleset):
            pattern = np.base_repr(i, k).zfill(2 * r + 1)
            rule_dict[pattern] = result
        return rule_dict

    @beartype
    def __call__(self, stage: NDArray) -> NDArray:
        """Call the cellular attomaton once on the stage
        Overwrites the input stage
        """
        padded_copy = np.pad(stage, [self.r, self.r])
        for i in range(len(stage)):
            stage[i] = self.rule_dict[self.get_part(padded_copy, i)]

        return stage

    @beartype
    def get_part(self, padded: NDArray, i: int) -> str:
        """Gets the i'th target of the stage
        Assumes the stage is padded (and therefore offset) according to self.r
        """
        target = padded[i : i + (2 * self.r + 1)]
        return "".join(map(str, target))
