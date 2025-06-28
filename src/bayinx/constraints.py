from typing import Tuple

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PyTree, Scalar, ScalarLike

from bayinx.core import Constraint, Parameter
from bayinx.core._parameter import T


class Lower(Constraint):
    """
    Enforces a lower bound on the parameter.
    """

    lb: Scalar

    def __init__(self, lb: ScalarLike):
        # assert greater than 1
        self.lb = jnp.asarray(lb)

    def constrain(self, param: Parameter[T]) -> Tuple[Parameter[T], Scalar]:
        """
        Enforces a lower bound on the parameter and adjusts the posterior density.

        # Parameters
        - `param`: The unconstrained `Parameter`.

        # Returns
        A tuple containing:
            - A modified `Parameter` with relevant leaves satisfying the constraint.
            - A scalar Array representing the log-absolute-Jacobian of the transformation.
        """
        # Extract relevant parameters(all inexact Arrays)
        dyn, static = eqx.partition(param, param.filter_spec)

        # Compute Jacobian adjustment
        total_laj: Scalar = jt.reduce(lambda a, b: a + b, jt.map(jnp.sum, dyn))

        # Compute transformation
        dyn = jt.map(lambda v: jnp.exp(v) + self.lb, dyn)

        # Combine into full parameter object
        param = eqx.combine(dyn, static)

        return param, total_laj


class LogSimplex(Constraint):
    """
    Enforces a log-transformed simplex constraint on the parameter.

    # Attributes
    - `sum`: The total sum of the parameter.
    """

    sum: Scalar

    def __init__(self, sum_val: ScalarLike = 1.0):
        """
        # Parameters
        - `sum_val`: The target sum for the exponentiated simplex. Defaults to 1.0.
        """
        self.sum = jnp.asarray(sum_val)

    def constrain(self, param: Parameter[T]) -> Tuple[Parameter[T], Scalar]:
        """
        Enforces a log-transformed simplex constraint on the parameter and adjusts the posterior density.

        # Parameters
        - `param`: The unconstrained `Parameter`.

        # Returns
        A tuple containing:
            - A modified `Parameter` with relevant leaves satisfying the constraint.
            - A scalar Array representing the log-absolute-Jacobian of the transformation.
        """
        # Partition the parameter into dynamic (to be transformed) and static parts
        dyn, static = eqx.partition(param, param.filter_spec)

        # Map transformation leaf-wise
        transformed = jt.map(self._transform_leaf, dyn) ## filter spec handles subsetting arrays, is_leaf unnecessary

        # Extract constrained parameters and Jacobian adjustments
        dyn_constrained: PyTree = jt.map(lambda x: x[0], transformed)
        lajs: PyTree = jt.map(lambda x: x[1], transformed)

        # Sum to get total Jacobian adjustment
        total_laj = jt.reduce(lambda a, b: a + b, lajs)

        # Recombine the transformed dynamic parts with the static parts
        param = eqx.combine(dyn_constrained, static)

        return param, total_laj

    def _transform_leaf(self, x: Array) -> Tuple[Array, Scalar]:
        """
        Internal function that applies a log-transformed simplex constraint on a single array.
        """
        laj: Scalar = jnp.array(0.0)

        # Save output shape
        output_shape: tuple[int, ...] = x.shape

        if x.size == 1:
            return(jnp.full(output_shape, jnp.log(self.sum)), laj)
        else:
            # Flatten x
            x = x.flatten()

            # Subset first K - 1 elements
            x = x[:-1]

            # Compute shifted cumulative sum
            zeta: Array = jnp.concat([jnp.zeros(1), x.cumsum()[:-1]])

            # Compute intermediate proportions vector
            eta: Array = jnn.sigmoid(x - zeta)

            # Compute Jacobian adjustment
            laj += jnp.sum(jnp.log(eta) + jnp.log(1 - eta)) # TODO: check for correctness

            # Compute log-transformed simplex weights
            w: Array = jnp.log(eta) + jnp.concatenate([jnp.array([0.0]), jnp.log(jnp.cumprod((1-eta)[:-1]))])
            w = jnp.concatenate([w, jnp.log(jnp.prod(1 - eta, keepdims=True))])

            # Scale unit simplex on log-scale
            w = w + jnp.log(self.sum)

            # Reshape for output
            w = w.reshape(output_shape)

            return (w, laj)
