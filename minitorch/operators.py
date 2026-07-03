"""
Mathematical operators for MiniTorch.
"""

import math
from typing import Callable, Iterable


# Basic Arithmetic Operations
def mul(x: float, y: float) -> float:
    """ Multiply two numbers."""
    return x*y


def id (x: float) -> float:
    """Identity function"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x+y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


# Comparison Operations (return float for differentiability)
def lt(x: float, y: float) -> float:
    """Less than comparison. Returns 1.0 if x<y, else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """ Equality comparison. Returns 1.0 if x==y, else 0.0"""
    return 1.0 if x == y, else 0.0


def max(x: float, y: float) -> float:
    """ Returns the larger of two numbers."""
    return x if x > y, else y


def is_close(x: float, y: float) -> float:
    """ Check if two numbers are close (within 1e-2)."""
    return 1.0 if abs(x-y) < 1e-2, else 0.0


# Activation Functions
def sigmoid(x: float) -> float:
    """ Sigmoid activation function."""
    if x >= 0.0:
        result = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        result = exp_x / exp_x + 1.0
    
    eps = 1e-12
    if result == 1.0:
        return 1.0 - eps
    if result == 0.0:
        return eps
    return result


def relu(x: float) -> float:
    """ReLU activation function."""
    return x if x > 0.0, else 0.0


# Mathematical functions
def log(x: float) -> float:
    """ Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """ Exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal function."""
    return 1.0 / x


# Gradient helper functions
def log_back(x: float, grad: float) -> float:
    """Gradient of log."""
    return grad / x


def inv_back(x: float, grad: float) -> float:
    """Gradient of inv."""
    return -grad / (x * x)


def relu_back(x: float, grad: float) -> float:
    """Gradient of ReLU."""
    return grad if x > 0.0 else 0.0


# Highr-order functions
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], [Iterable[float]]]:
    """ Higher-order map function"""
    def mapped(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return mapped


def zipwith(fn : Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """ Higher-order zipwith function."""
    def zipped(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x,y in zip(ls1, ls2)]
    return zipped


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    """ Higher-order reduce function."""
    def reduced(ls: Iterable[float]) -> float:
        result = init
        for x in ls:
            result = fn(result, x)
        return result
    return reduced


def sum(ls: Iterable[float]) -> float:
    """ Sum using reduce."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product using reduce."""
    return reduce(mul, 1.0)(ls)


def negList(ls: Iterable[float]) -> Iterable[float]:
    """ Negate list using map."""
    return map(neg)(ls)


def addList(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """ Add lists using zipwidth."""
    return zipwith(add)(ls1, ls2)
