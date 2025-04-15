from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Scalar, ScalarLike


class Constraint(eqx.Module):
    """
    Abstract base class for defining parameter constraints.

    Subclasses should implement the `constrain` method to apply the
    transformation and compute the ladj adjustment.
    """
    @abstractmethod
    def constrain(self, x: ArrayLike) -> Tuple[Array, Scalar]:
        """
        Applies the constraining transformation to an unconstrained input
        and computes the log absolute determinant of the Jacobian (ladj)
        of this transformation.

        # Parameters
        - `x`: The unconstrained JAX Array-like input.

        # Returns
        A tuple containing:
            - The constrained JAX Array.
            - A scalar JAX Array representing the ladj of the transformation.
        """
        pass


class LowerBound(Constraint):
    """
    Enforces a lower bound on the parameter.
    """
    lb: ScalarLike

    def __init__(self, lb: ScalarLike):
        self.lb = lb

    def constrain(self, x: ArrayLike) -> Tuple[Array, Scalar]:
        """
        Applies the lower bound constraint and computes the ladj.

        # Parameters
        - `x`: The unconstrained JAX Array-like input.

        # Parameters
        A tuple containing:
            - The constrained JAX Array (x > self.lb).
            - A scalar JAX Array representing the ladj of the transformation.
        """
        # Compute transformation adjustment
        ladj = jnp.sum(x)

        # Compute transformation
        x = jnp.exp(x) + self.lb

        return x, ladj
