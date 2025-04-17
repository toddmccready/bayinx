from abc import abstractmethod
from typing import Tuple

import equinox as eqx
from jaxtyping import Array, ArrayLike, Scalar


class Constraint(eqx.Module):
    """
    Abstract base class for defining parameter constraints.
    """

    @abstractmethod
    def constrain(self, x: ArrayLike) -> Tuple[Array, Scalar]:
        """
        Applies the constraining transformation to an unconstrained input and computes the log-absolute-Jacobian of the transformation.

        # Parameters
        - `x`: The unconstrained JAX Array-like input.

        # Returns
        A tuple containing:
            - The constrained JAX Array.
            - A scalar JAX Array representing the log-absolute-Jacobian of the transformation.
        """
        pass
