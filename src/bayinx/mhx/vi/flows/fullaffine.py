from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar

from bayinx.core import Flow


class FullAffine(Flow):
    """
    A full affine flow.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the scale and shift parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    def __init__(self, dim: int):
        """
        Initializes a full affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.params = {
            "shift": jnp.zeros(dim),
            "scale": jnp.zeros((dim, dim)),
        }

        if dim == 1:
            self.constraints = {}
        else:

            @eqx.filter_jit
            def constrain_scale(scale: Array):
                # Extract diagonal and apply exponential
                diag: Array = jnp.exp(jnp.diag(scale))

                # Return matrix with modified diagonal
                return jnp.fill_diagonal(scale, diag, inplace=False)

            self.constraints = {"scale": constrain_scale}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        params = self.transform_pars()

        # Extract parameters
        shift: Array = params["shift"]
        scale: Array = params["scale"]

        # Compute forward transformation
        draws = draws @ scale + shift

        return draws

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def adjust_density(self, draws: Array) -> Tuple[Array, Scalar]:
        params = self.transform_pars()

        # Extract parameters
        shift: Array = params["shift"]
        scale: Array = params["scale"]

        # Compute forward transformation
        draws = draws @ scale + shift

        # Compute laj
        laj: Scalar = jnp.log(jnp.diag(scale)).sum()

        return draws, laj
