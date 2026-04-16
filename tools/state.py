from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class TrussState:
    K: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    nodes: list = field(default_factory=list)
    members: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    n_dof: int = 0
    u: np.ndarray = field(default_factory=lambda: np.empty(0))
    F: np.ndarray = field(default_factory=lambda: np.empty(0))
    reactions: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_analysis: list = field(default_factory=list)
    ready: bool = False      # True after build_truss
    solved: bool = False     # True after solve_truss


@dataclass
class BeamState:
    last_result: dict = field(default_factory=dict)


truss_state = TrussState()
beam_state = BeamState()