import numpy as np
from itertools import chain, combinations
from enum import IntEnum


class NodeAttr(IntEnum):
    PARENT = 0
    CHILDREN = 1
    VISITS = 2
    VALUE = 3
    BELIEFS = 4


def ucb(N, n, V, c=1):
    """
    Compute the UCB score
    :param N: number of visits of the parent
    :param n: number of visits of the child
    :param V: the child's value
    :param c: the exploration constant
    :return: the UCB value.
    """
    return float('inf') if n == 0 else V + c * np.sqrt(np.log(N) / n)


def powerset(iterable):
    """
    Compute the power set of the input set
    :param iterable: the input set
    :return: the power set of the input.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
