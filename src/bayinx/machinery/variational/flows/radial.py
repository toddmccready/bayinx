from functools import partial
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import norm
from jaxtyping import Array, Float

from bayinx.core import Flow


class Radial(Flow):
    """
    A radial flow.

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
            "alpha": jnp.array(1.0),
            "_beta": jnp.array(1.0),
            "center": jnp.ones(dim),
        }
        self.constraints = {"_beta": jnp.exp}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward radial transformation for each draw.

        # Parameters
        - `draws`: Draws from some layer of a normalizing flow.

        # Returns
        The transformed samples.
        """
        # Constrain parameters
        params = self.constrain()

        # Extract parameters
        alpha = params["alpha"]
        beta = params["_beta"] - params["alpha"]
        center = params["center"]

        # Compute distance to center per-draw
        r: Array = norm(draws - params["center"], axis=1)

        return draws + (beta / (alpha + r)) * (draws - center)

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
        h_prime = 1.0 - jnp.square(jnp.tanh(jnp.dot(draws, params["w"]) + params["b"]))

        return jnp.log(jnp.abs(1.0 + h_prime * params["u"].dot(params["w"])))
