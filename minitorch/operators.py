"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.
    Args:
        x, y: Input mumbers
    Returns:
        The product of x and y: x * y
    """
    return x * y


def id(x: float) -> float:
    """Identity function - returns the input unchanged.
    Args:
        x: Input number
    Returns:
        The input value x
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.
    Args:
        x, y: Input mumbers
    Returns:
        The sum of x and y: x + y
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.
    Args:
        x: Input number
    Returns:
        The negative of x: -x
    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.
    Args:
        x, y: Input numbers
    Returns:
        True if x < y, False otherwise
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.
    Args:
        x, y: Input numbers
    Returns:
        True if x == y, False otherwise
    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.
    Args:
        x, y: Input numbers
    Returns:
        The maximum of x and y
    """
    return x if x > y else y


def is_close(x: float, y: float, eps: float = 1e-2) -> bool:
    """Checks if two numbers are close in value.
    Args:
        x, y: Input numbers
        eps: Tolerance threshold (default: 1e-2) 
    Returns:
        True if |x - y| < eps, False otherwise
    """
    return abs(x - y) < eps


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.
    Uses numerically stable formula:
    f(x) = 1.0 / (1.0 + e^{-x}) if x >= 0 else e^x / (1.0 + e^x)
    Args:
        x: Input number
    Returns:
        Sigmoid activation value between 0 and 1
    """
    return 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.
    Args:
        x: Input number 
    Returns:
        x if x > 0, else 0
    """
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm.
    Args:
        x: Input number (must be positive) 
    Returns:
        Natural logarithm of x
    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function.
    Args:
        x: Input number 
    Returns:
        e raised to the power of x
    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second argument.
    If f(x) = log(x), computes d * f'(x) = d * (1/x)
    Args:
        x: Input to log function
        d: Second argument to multiply by derivative
    Returns:
        d * (1/x)
    """
    return y / x


def inv(x: float) -> float:
    """Calculates the reciprocal.
    Args:
        x: Input number (cannot be zero) 
    Returns:
        Reciprocal of x: 1/x
    """
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second argument.
    If f(x) = 1/x, computes d * f'(x) = d * (-1/x^2)
    Args:
        x: Input to reciprocal function
        d: Second argument to multiply by derivative
    Returns:
        d * (-1/x^2)
    """
    return -y / (x ** 2)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second argument.
    If f(x) = relu(x), computes d * f'(x) = d if x > 0 else 0
    Args:
        x: Input to ReLU function
        d: Second argument to multiply by derivative
    Returns:
        d if x > 0, else 0
    """
    return y if x > 0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], l: Iterable[float]) -> list:
    """Higher-order function that applies a given function to each element of an iterable.
    Args:
        fn: Function that takes one element and returns a transformed element
        l: Iterable of input elements
    Returns:
        New list with fn applied to each element of the input iterable
    """
    return [fn(x) for x in l]


def zipWith(fn: Callable[[float, float], float], l1: Iterable[float], l2: Iterable[float]) -> list:
    """Higher-order function that combines elements from two iterables using a given function.
    Args:
        fn: Function that combines two elements into one
        l1: First iterable of elements
        l2: Second iterable of elements (must be same length as l1)
    Returns:
        New list where each element is fn(a, b) for corresponding a in l1, b in l2
    """
    return [fn(a, b) for a, b in zip(l1, l2)]


def reduce(fn: Callable[[float, float], float], l: Iterable[float], start: float) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.
    Applies fn cumulatively from left to right: fn(...fn(fn(start, x1), x2)..., xn)
    Args:
        fn: Binary function that combines accumulator with current element
        l: Iterable of elements to reduce
        start: Initial accumulator value
    Returns:
        Single reduced value
    """
    result = start
    for i in l:
        result = fn(i, result)
    return result


def negList(l: Iterable[float]) -> list:
    """Negate all elements in a list using map.
    Args:
        l: Iterable of numbers to negate
    Returns:
        New list with each element negated
    """
    return map(neg, l)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> list:
    """Add corresponding elements from two lists using zipWith.
    Args:
        l1: First iterable of numbers
        l2: Second iterable of numbers (must be same length as l1)
    Returns:
        New list where each element is the sum of corresponding elements from l1 and l2
    """
    return zipWith(add, l1, l2)


def sum(l: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.
    Args:
        l: Iterable of numbers to sum
    Returns:
        Sum of all elements in the iterable
    """
    return reduce(add, l, 0.0)


def prod(l: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.
    Args:
        l: Iterable of numbers to multiply
    Returns:
        Product of all elements in the iterable
    """
    return reduce(mul, l, 1.0)
