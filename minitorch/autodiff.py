from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    args = list(vals)
    args[arg] = vals[arg] + epsilon
    f_plus = f(*args)

    args[arg] = vals[arg] - epsilon
    f_minus = f(*args)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result = []
    
    def dfs(node: Variable) -> None:
        """
        Depth-first search that visits nodes in post-order.
        This ensures parents are processed after their children.
        """
        if node.unique_id in visited or node.is_constant():
            return

        visited.add(node.unique_id)
        if hasattr(node, 'parents') and node.parents:
            for parent in node.parents:
                if not parent.is_constant():
                    dfs(parent)
        result.append(node)
    
    dfs(variable)
    reversed_result = list(reversed(result))
    return reversed_result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    
    for node in order:
        current_deriv = derivatives.get(node.unique_id, 0.0)
        
        if node.is_leaf():
            node.accumulate_derivative(current_deriv)  # If this is a leaf node, accumulate the derivative
        else:
            parent_derivs = node.chain_rule(current_deriv)  # Else apply chain rule to get derivatives for parents
            
    
            for parent, local_deriv in parent_derivs:
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0.0
                derivatives[parent.unique_id] += local_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
