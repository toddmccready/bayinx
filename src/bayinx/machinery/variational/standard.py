from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Key

from bayinx.core import Model, Variational
from bayinx.dists import normal


class Standard(Variational):
    """
    A standard normal distribution approximation to a posterior distribution.

    # Attributes
    - `dim`: Dimension of the parameter space.
    """

    dim: int = eqx.field(static=True)
    _unflatten: Callable[[Float[Array, "..."]], Model] = eqx.field(static=True)
    _constraints: Model = eqx.field(static=True)

    def __init__(self, model: Model):
        """
        Constructs a standard normal approximation to a posterior distribution.

        # Parameters
        - `model`: A probabilistic `Model` object.
        """
        # Partition model
        _, self._constraints = eqx.partition(model, eqx.is_array)

        # Flatten params component
        _, self._unflatten = ravel_pytree(_)

        # Store dimension of parameter space
        self.dim = jnp.size(_)

    @eqx.filter_jit
    def sample(self, n: int, key: Key = jr.PRNGKey(0)) -> Array:
        # Sample variational draws
        draws: Array = jr.normal(key=key, shape=(n, self.dim))

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return normal.logprob(
            x=draws,
            mu=jnp.array(0.0),
            sigma=jnp.array(1.0),
        ).sum(axis=1)

    @eqx.filter_jit
    def filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)

        return filter_spec

    def elbo(self):
        return None
