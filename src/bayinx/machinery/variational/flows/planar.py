from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Scalar

from bayinx.core import Flow


class Planar(Flow):
    """
    A planar flow.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the flow parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]] = (
        eqx.field(static=True)
    )

    def __init__(self, dim: int, key=jr.PRNGKey(0)):
        """
        Initializes a planar flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.params = {
            "u": jr.normal(key, (dim,)),
            "w": jr.normal(key, (dim,)),
            "b": jr.normal(key, (1,)),
        }
        self.constraints = {}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        params = self.constrain()

        # Extract parameters
        w: Array = params["w"]
        u: Array = params["u"]
        b: Array = params["b"]

        # Compute forward transformation
        draws = draws + u * jnp.tanh(draws.dot(w) + b)

        return draws

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def transform(self, draws: Array) -> Tuple[Scalar, Array]:
        params = self.constrain()

        # Extract parameters
        w: Array = params["w"]
        u: Array = params["u"]
        b: Array = params["b"]

        # Compute shared intermediates
        x: Array = draws.dot(w) + b

        # Compute forward transformation
        draws = draws + u * jnp.tanh(x)

        # Compute ladj
        h_prime: Scalar = 1.0 - jnp.square(jnp.tanh(x))
        ladj: Scalar = jnp.log(jnp.abs(1.0 + h_prime * u.dot(w)))

        return ladj, draws
