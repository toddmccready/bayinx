from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Scalar, ScalarLike

from bayinx.core import Constraint, Parameter
from bayinx.core._parameter import T


class Lower(Constraint):
    """
    Enforces a lower bound on the parameter.
    """

    lb: Scalar

    def __init__(self, lb: ScalarLike):
        self.lb = jnp.array(lb)

    def constrain(self, param: Parameter[T]) -> Tuple[Parameter[T], Scalar]:
        """
        Enforces a lower bound on the parameter and adjusts the posterior density.

        # Parameters
        - `param`: The unconstrained `Parameter`.

        # Parameters
        A tuple containing:
            - A modified `Parameter` with relevant leaves satisfying the constraint.
            - A scalar Array representing the log-absolute-Jacobian of the transformation.
        """
        # Extract relevant parameters(all Array)
        dyn, static = eqx.partition(param, param.filter_spec)

        # Compute density adjustment
        laj: Scalar = jt.reduce(lambda a, b: a + b, jt.map(jnp.sum, dyn))

        # Compute transformation
        dyn = jt.map(lambda v: jnp.exp(v) + self.lb, dyn)

        # Combine into full parameter object
        param = eqx.combine(dyn, static)

        return param, laj
