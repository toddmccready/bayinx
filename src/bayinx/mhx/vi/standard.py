import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Key

from bayinx.core._variational import M, Variational
from bayinx.dists import normal


class Standard(Variational[M]):
    """
    A standard normal approximation to a posterior distribution.

    # Attributes
    - `dim`: Dimension of the parameter space.
    """

    dim: int

    def __init__(self, model: M):
        """
        Constructs a standard normal approximation to a posterior distribution.

        # Parameters
        - `model`: A probabilistic `Model` object.
        """
        # Partition model
        params, self._constraints = eqx.partition(model, model.filter_spec)

        # Flatten params component
        params, self._unflatten = ravel_pytree(params)

        # Store dimension of parameter space
        self.dim = jnp.size(params)

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
        ).sum(axis=1, keepdims=True)

    @property
    def filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)

        return filter_spec

    def elbo(self):
        return None

    def elbo_grad(self):
        return None
