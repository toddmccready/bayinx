from functools import partial
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from bayinx.core import Flow


class Planar(Flow):
    """
    A Planar flow diffeomorphism.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the flow parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]] = (
        eqx.field(static=True)
    )

    def __init__(self, dim: int, key = jr.PRNGKey(0)):
        """
        Initializes a Planar flow diffeomorphism.

        # Parameters
        - `dim`: The dimension of the parameter space of interest.
        """
        self.params = {
            "u": jr.normal(key, (dim,)),
            "w": jr.normal(key, (dim,)),
            "b": jr.normal(key, (1,)),
        }
        self.constraints = {}  # Consider constraints for invertibility

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward planar transformation for each draw.

        # Parameters
        - `draws`: Draws from some layer of a normalizing flow.

        # Returns
        The transformed samples.
        """
        params = self.constrain()
        activation = jnp.tanh(jnp.dot(draws, params["w"]) + params["b"])
        return draws + params["u"] * activation[:, None]

    @partial(jax.vmap, in_axes=(None, 0))
    @eqx.filter_jit
    def ladj(self, draws: Array) -> Array:
        """
        Computes the log-absolute-determinant of the Jacobian of the forward transformation for each draw.

        # Parameters
        - `draws`: Draws from some layer of a normalizing flow.

        # Returns
        The log-absolute-determinant of the Jacobian per-draw.
        """
        params = self.constrain()

        # Compute derivative of nonlinear function
        h_prime = 1 - jnp.square(jnp.tanh(jnp.dot(draws, params["w"]) + params["b"]))

        return jnp.log(jnp.abs(1 + jnp.dot(params["u"], params["w"]) * h_prime))
