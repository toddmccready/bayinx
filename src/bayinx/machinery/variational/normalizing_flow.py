from typing import Any, Callable, Self, Tuple

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Key, Scalar

from bayinx.core import Flow, Model, Variational


class NormalizingFlow(Variational):
    """
    An ordered collection of diffeomorphisms that map a base distribution to a
    normalized approximation of a posterior distribution.

    # Attributes
    - `base`: A base variational distribution.
    - `flows`: An ordered collection of continuously parameterized
        diffeomorphisms.
    """

    flows: list[Flow]
    base: Variational
    _unflatten: Callable[[Float[Array, "..."]], Model] = eqx.field(static=True)
    _constraints: Model = eqx.field(static=True)

    def __init__(self, base: Variational, flows: list[Flow], model: Model):
        """
        Constructs an unoptimized normalizing flow posterior approximation.

        # Parameters
        - `base`: The base variational distribution.
        - `flows`: A list of diffeomorphisms.
        - `model`: A probabilistic `Model` object.
        """
        # Partition model
        params, self._constraints = eqx.partition(model, eqx.is_array)

        # Flatten params component
        flat_params, self._unflatten = jfu.ravel_pytree(params)

        self.base = base
        self.flows = flows

    @eqx.filter_jit
    def sample(self, n: int, key: Key = jr.PRNGKey(0)):
        """
        Sample from the variational distribution `n` times.
        """
        # Sample from the base distribution
        draws: Array = self.base.sample(n, key)

        # Apply forward transformations
        for map in self.flows:
            draws = map.forward(draws)

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        # Evaluate base density
        variational_evals: Array = self.base.eval(draws)

        for map in self.flows:
            # Compute adjustment
            ladj, draws = map.adjust_density(draws)

            # Adjust variational density
            variational_evals = variational_evals - ladj

        return variational_evals

    @eqx.filter_jit
    def _eval(self, draws: Array, data=None) -> Tuple[Scalar, Array]:
        """
        Evaluate the posterior and variational densities at the transformed
        `draws` to avoid extra compute when requiring variational draws for
        the posterior evaluation.

        # Parameters
        - `draws`: Draws from the base variational distribution.
        - `data`: Any data required to evaluate the posterior density.

        # Returns
        The posterior and variational densities.
        """
        # Evaluate base density
        variational_evals: Array = self.base.eval(draws)

        for map in self.flows:
            # Compute adjustment
            ladj, draws = map.adjust_density(draws)

            # Adjust variational density
            variational_evals = variational_evals - ladj

        # Evaluate posterior at final variational draws
        posterior_evals = self.eval_model(draws, data)

        return posterior_evals, variational_evals

    def filter_spec(self):
        # Only optimize the parameters of the flows
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda nf: nf.flows,
            filter_spec,
            replace=True,
        )

        return filter_spec

    @eqx.filter_jit
    def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
        # Partition
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: Key, data: Any = None):
            # Combine
            self = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = self.base.sample(n, key)

            posterior_evals, variational_evals = self._eval(draws, data)
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
            self = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = self.base.sample(n, key)

            posterior_evals, variational_evals = self._eval(draws, data)
            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo_grad(dyn, n, key, data)
