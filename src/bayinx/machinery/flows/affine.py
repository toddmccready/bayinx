from functools import partial
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bayinx.core import Flow


class Affine(Flow):
    """
    An affine diffeomorphism.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the scale and shift parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]] = (
        eqx.field(static=True)
    )

    def __init__(self, dim: int):
        """
        Initializes an affine diffeomorphism.

        # Parameters
        - `dim`: The dimension of the parameter space of interest.
        """
        self.params = {
            "shift": jnp.zeros(dim),
            "scale": jnp.zeros((dim, dim)),
        }

        self.constraints = {"scale": lambda m: jnp.tril(jnp.exp(m))}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward affine transformation for each draw.

        # Parameters
        - `draws`: Draws from some layer of a normalizing flow.

        # Returns
        The transformed samples.
        """
        params = self.constrain()

        return draws @ params["scale"] + params["shift"]

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
        return jnp.log(jnp.diag(params["scale"])).sum()
