from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TrussState:
    K: Optional[np.ndarray] = None
    nodes: Optional[list] = None
    members: Optional[list] = None
    supports: Optional[list] = None
    n_dof: Optional[int] = None
    u: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    reactions: Optional[np.ndarray] = None
    last_analysis: Optional[list] = None


@dataclass
class BeamState:
    last_result: Optional[dict] = None


# singletons — import these everywhere
truss_state = TrussState()
beam_state = BeamState()