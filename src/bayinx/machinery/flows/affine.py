from functools import partial
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bayinx.core import Flow


class ElementwiseAffine(Flow):
    """
    An elementwise affine diffeomorphism.

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
        Initializes an elementwise affine diffeomorphism.

        # Parameters
        - `dim`: The dimension of the parameter space of interest.
        """
        self.params = {
            "shift": jnp.repeat(jnp.array(0.0), dim),
            "scale": jnp.repeat(jnp.array(0.0), dim),
        }
        self.constraints = {"scale": jnp.exp}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward elementwise affine transformation for each draw.

        # Parameters
        - `draws`: A collection of variational draws.

        # Returns
        The transformed samples.
        """
        params = self.constrain()

        return draws * params["scale"] + params["shift"]

    @partial(jax.vmap, in_axes=(None, 0))
    @eqx.filter_jit
    def ladj(self, draws: Array) -> Array:
        """
        Computes the log-absolute-determinant of the Jacobian for each draw of the reverse transformation.

        # Parameters
        - `draws`: Variational draws.

        # Returns
        The log-absolute-determinant of the Jacobian.
        """

        params = self.constrain()
        return jnp.log(params["scale"]).sum()
