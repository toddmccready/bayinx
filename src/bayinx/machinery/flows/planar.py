from functools import partial
from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from bayinx.core import Flow


class Planar(Flow):
    """
    An elementwise affine diffeomorphism.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the scale and shift parameters.
    - `constraints`: A dictionary of constraining transformations.
    - `transform`: An elementwise monotonic
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]] = (
        eqx.field(static=True)
    )
    smooth_map: Callable[[Scalar], Scalar] = eqx.field(static=True)
    smooth_der: Callable[[Scalar], Scalar] = eqx.field(static=True)

    def __init__(self, dim: int, smooth_map: Callable[[Scalar], Scalar] = lambda x: x):
        """
        Initializes an elementwise affine diffeomorphism.

        # Parameters
        - `dim`: The dimension of the parameter space of interest.
        """
        self.params = {
            "w": jnp.repeat(jnp.array(0.0), dim),
            "b": jnp.array(0.0),
            "u": jnp.repeat(jnp.array(0.0), dim),
        }
        self.constraints = {}

        self.smooth_map = smooth_map
        self.smooth_der = jax.grad(jax.jit(smooth_map))

    @eqx.filter_jit
    def forward(self, draws: Array) -> Array:
        """
        Applies the forward transformation for each draw.

        # Parameters
        - `draws`: A collection of variational draws.

        # Returns
        The transformed samples.
        """
        params = self.constrain()

        return draws * params["scale"] + params["shift"]

    @eqx.filter_jit
    def reverse(self, draws: Array) -> Array:
        """
        Applies the reverse elementwise affine transformation for each draw.

        # Parameters
        - `draws`: A collection of variational draws.

        # Returns
        The transformed samples.
        """
        params = self.constrain()

        return (draws - params["shift"]) / params["scale"]

    @partial(jax.vmap, in_axes=(None, 0))
    @eqx.filter_jit
    def inverse_ladj(self, draws: Array) -> Array:
        """
        Computes the log-absolute-determinant of the Jacobian for each draw of the reverse transformation.

        # Parameters
        - `draws`: A collection of variational draws.

        # Returns
        The log-absolute-determinant of the Jacobian.
        """

        params = self.constrain()
        return -jnp.log(params["scale"]).sum()
