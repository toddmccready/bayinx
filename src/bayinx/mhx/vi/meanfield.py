from typing import Any, Generic, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Key, Scalar

from bayinx.core import Model, Variational
from bayinx.dists import normal

M = TypeVar("M", bound=Model)


class MeanField(Variational, Generic[M]):
    """
    A fully factorized Gaussian approximation to a posterior distribution.

    # Attributes
    - `mean`: The mean of the unconstrained approximation.
    - `log_std` The log-transformed standard deviation of the unconstrained approximation.
    """

    mean: Array
    log_std: Array

    def __init__(self, model: M, init_log_std: float = -5.0):
        """
        Constructs an unoptimized meanfield posterior approximation.

        # Parameters
        - `model`: A probabilistic `Model` object.
        - `init_log_std`: The initial log-transformed standard deviation of the Gaussian approximation.
        """
        # Partition model
        params, self._static = eqx.partition(model, model.filter_spec)

        # Flatten params component
        params, self._unflatten = ravel_pytree(params)

        # Initialize variational parameters
        self.mean = params
        self.log_std = jnp.full(params.size, init_log_std, params.dtype)

    @property
    @eqx.filter_jit
    def filter_spec(self):
        # Generate empty specification
        filter_spec = jtu.tree_map(lambda _: False, self)

        # Specify variational parameters
        filter_spec = eqx.tree_at(
            lambda mf: mf.mean,
            filter_spec,
            replace=True,
        )
        filter_spec = eqx.tree_at(
            lambda mf: mf.log_std,
            filter_spec,
            replace=True,
        )

        return filter_spec

    @eqx.filter_jit
    def sample(self, n: int, key: Key = jr.PRNGKey(0)) -> Array:
        # Sample variational draws
        draws: Array = (
            jr.normal(key=key, shape=(n, self.mean.size))
            * jnp.exp(self.log_std)
            + self.mean
        )

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return normal.logprob(
            x=draws,
            mu=self.mean,
            sigma=jnp.exp(self.log_std),
        ).sum(axis=1)

    @eqx.filter_jit
    def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: Key, data: Any = None) -> Scalar:
            vari = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = vari.sample(n, key)

            # Evaluate posterior density for each draw
            posterior_evals: Array = vari.eval_model(draws, data)

            # Evaluate variational density for each draw
            variational_evals: Array = vari.eval(draws)

            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo(dyn, n, key, data)

    @eqx.filter_jit
    def elbo_grad(self, n: int, key: Key, data: Any = None) -> Self:
        dyn, static = eqx.partition(self, self.filter_spec)

        @eqx.filter_jit
        @eqx.filter_grad
        def elbo_grad(dyn: Self, n: int, key: Key, data: Any = None):
            vari = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = vari.sample(n, key)

            # Evaluate posterior density for each draw
            posterior_evals: Array = vari.eval_model(draws, data)

            # Evaluate variational density for each draw
            variational_evals: Array = vari.eval(draws)

            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo_grad(dyn, n, key, data)
