import numpy as np 

## define a function to calculate the entropy in a subset of the data
def entropy(data: np.array) -> float:
    """
    Calculates the entropy of a dataset.
    The entropy is the sum of p(x)log(p(x)) across all the different possible outcomes.
    The higher the entropy, the more "random" the data is.
    """
    from math import log
    from collections import Counter
    counts = Counter(data)
    probs = [counts[x] / len(data) for x in counts]
    return -sum([p * log(p, 2) for p in probs])

## define a function to calculate the information gain from the left and right branches
def info_gain(left: np.array, right: np.array, current_uncertainty: float) -> float:
    """
    Calculates the information gain.
    """
    p = len(left) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

