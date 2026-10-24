"""Type stubs for scipy.special module."""

import numpy as np
from numpy.typing import ArrayLike

def expit(x: ArrayLike) -> float | np.ndarray: ...
def logit(x: ArrayLike) -> float | np.ndarray: ...
