from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from bayinx.core import Flow


class Affine(Flow):
    """
    An affine flow.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the scale and shift parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]]

    def __init__(self, dim: int):
        """
        Initializes an affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.params = {
            "shift": jnp.zeros(dim),
            "scale": jnp.zeros((dim, dim)),
        }

        self.constraints = {"scale": lambda m: jnp.tril(jnp.exp(m))}

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        params = self.constrain_pars()

        # Extract parameters
        shift: Array = params["shift"]
        scale: Array = params["scale"]

        # Compute forward transformation
        draws = draws @ scale + shift

        return draws

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def adjust_density(self, draws: Array) -> Tuple[Scalar, Array]:
        params = self.constrain_pars()

        # Extract parameters
        shift: Array = params["shift"]
        scale: Array = params["scale"]

        # Compute forward transformation
        draws = draws @ scale + shift

        # Compute ladj
        ladj: Scalar = jnp.log(jnp.diag(scale)).sum()

        return ladj, draws
