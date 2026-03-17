import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x,p=np.array(x),np.array(p)
    if len(x) != len(p):
        raise ValueError("Arrays 'x' and 'p' must have the same length.")

    if np.any(p < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to exactly 1.")

    return x@p
