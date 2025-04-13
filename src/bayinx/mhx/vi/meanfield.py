from typing import Any, Callable, Dict, Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Key, Scalar

from bayinx.core import Model, Variational
from bayinx.dists import normal


class MeanField(Variational):
    """
    A fully factorized Gaussian approximation to a posterior distribution.

    # Attributes
    - `var_params`: The variational parameters for the approximation.
    """

    var_params: Dict[str, Float[Array, "..."]]
    _unflatten: Callable[[Float[Array, "..."]], Model] = eqx.field(static=True)
    _constraints: Model = eqx.field(static=True)

    def __init__(self, model: Model):
        """
        Constructs an unoptimized meanfield posterior approximation.

        # Parameters
        - `model`: A probabilistic `Model` object.
        """
        # Partition model
        params, self._constraints = eqx.partition(model, model.filter_spec())

        # Flatten params component
        params, self._unflatten = ravel_pytree(params)

        # Initialize variational parameters
        self.var_params = {
            "mean": params,
            "log_std": jnp.zeros(params.size, dtype=params.dtype),
        }

    @eqx.filter_jit
    def sample(self, n: int, key: Key = jr.PRNGKey(0)) -> Array:
        # Sample variational draws
        draws: Array = (
            jr.normal(key=key, shape=(n, self.var_params["mean"].size))
            * jnp.exp(self.var_params["log_std"])
            + self.var_params["mean"]
        )

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return normal.logprob(
            x=draws,
            mu=self.var_params["mean"],
            sigma=jnp.exp(self.var_params["log_std"]),
        ).sum(axis=1)

    @eqx.filter_jit
    def filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda mf: mf.var_params,
            filter_spec,
            replace=True,
        )
        return filter_spec

    @eqx.filter_jit
    def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
        """
        Estimate the ELBO and its gradient(w.r.t the variational parameters).
        """
        # Partition variational
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: Key, data: Any = None) -> Scalar:
            # Combine
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
        # Partition
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_grad
        @eqx.filter_jit
        def elbo_grad(dyn: Self, n: int, key: Key, data: Any = None):
            # Combine
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
