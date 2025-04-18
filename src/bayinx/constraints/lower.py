from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar, ScalarLike

from bayinx.core.constraint import Constraint
from bayinx.core.parameter import Parameter


class Lower(Constraint):
    """
    Enforces a lower bound on the parameter.
    """

    lb: Scalar

    def __init__(self, lb: ScalarLike):
        self.lb = jnp.array(lb)

    @eqx.filter_jit
    def constrain(self, x: Parameter) -> Tuple[Parameter, Scalar]:
        """
        Enforces a lower bound on the parameter and adjusts the posterior density.

        # Parameters
        - `x`: The unconstrained `Parameter`.

        # Parameters
        A tuple containing:
            - A modified `Parameter` with relevant leaves satisfying the constraint.
            - A scalar Array representing the log-absolute-Jacobian of the transformation.
        """
        # Extract relevant filter specification
        filter_spec = x.filter_spec

        # Extract relevant parameters(all Array)
        dyn_params, static_params = eqx.partition(x, filter_spec)

        # Compute density adjustment
        laj: PyTree = jt.map(jnp.sum, dyn_params) # pyright: ignore
        laj: Scalar = jt.reduce(lambda a,b: a + b, laj)

        # Compute transformation
        dyn_params = jt.map(lambda v: jnp.exp(v) + self.lb, dyn_params)

        # Combine into full parameter object
        x = eqx.combine(dyn_params, static_params)

        return x, laj
