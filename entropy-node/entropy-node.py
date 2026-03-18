import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    y=np.array(y)
    elements,counts=np.unique(y,return_counts=True)
    probs = counts / np.sum(counts)
    probs=np.clip(probs,1e-15,1)
    h=-1* np.sum(probs*np.log2(probs))
    return h