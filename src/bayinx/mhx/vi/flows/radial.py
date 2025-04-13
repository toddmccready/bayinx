from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import norm
from jaxtyping import Array, Float, Scalar

from bayinx.core import Flow


class Radial(Flow):
    """
    A radial flow.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the flow parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]]

    def __init__(self, dim: int, key=jr.PRNGKey(0)):
        """
        Initializes a planar flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.params = {
            "alpha": jnp.array(1.0),
            "beta": jnp.array(1.0),
            "center": jnp.ones(dim),
        }
        self.constraints = {"beta": jnp.exp}

    @partial(jax.vmap, in_axes=(None, 0))
    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward radial transformation for each draw.

        # Parameters
        - `draws`: Draws from some layer of a normalizing flow.

        # Returns
        The transformed samples.
        """
        params = self.transform_pars()

        # Extract parameters
        alpha = params["alpha"]
        beta = params["beta"]
        center = params["center"]

        # Compute distance to center per-draw
        r: Array = norm(draws - center)

        # Apply forward transformation
        draws = draws + (beta / (alpha + r)) * (draws - center)

        return draws

    @partial(jax.vmap, in_axes=(None, 0))
    @eqx.filter_jit
    def adjust_density(self, draws: Array) -> Tuple[Scalar, Array]:
        params = self.transform_pars()

        # Extract parameters
        alpha = params["alpha"]
        beta = params["beta"]
        center = params["center"]

        # Compute distance to center per-draw
        r: Array = norm(draws - center)

        # Compute shared intermediates
        x: Array = beta / (alpha + r)

        # Apply forward transformation
        draws = draws + (x) * (draws - center)

        # Compute density adjustment
        ladj = jnp.log(
            jnp.abs(
                (1.0 + alpha * beta / (alpha + r) ** 2.0)
                * (1.0 + x) ** (center.size - 1.0)
            )
        )

        return ladj, draws
