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
    _unflatten: Callable[[Float[Array, "..."]], Model]
    _constraints: Model

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
        _, self._unflatten = jfu.ravel_pytree(params)

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
    def __eval(self, draws: Array, data=None) -> Tuple[Array, Array]:
        """
        Evaluate the posterior and variational densities at the transformed
        `draws` to avoid extra compute when requiring variational draws for
        the posterior evaluation.

        # Parameters
        - `draws`: Draws from the base variational distribution.
        - `data`: Any data required to evaluate the posterior density.

        # Returns
        The posterior and variational densities as JAX Arrays.
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
        # Generate empty specification
        filter_spec = jtu.tree_map(lambda _: False, self)

        # Specify variational parameters based on each flow's filter spec.
        filter_spec = eqx.tree_at(
            lambda vari: vari.flows,
            filter_spec,
            replace=[flow.filter_spec() for flow in self.flows],
        )

        return filter_spec

    @eqx.filter_jit
    def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: Key, data: Any = None):
            self = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = self.base.sample(n, key)

            posterior_evals, variational_evals = self.__eval(draws, data)
            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo(dyn, n, key, data)

    @eqx.filter_jit
    def elbo_grad(self, n: int, key: Key, data: Any = None) -> Self:
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_grad
        @eqx.filter_jit
        def elbo_grad(dyn: Self, n: int, key: Key, data: Any = None):
            self = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = self.base.sample(n, key)

            posterior_evals, variational_evals = self.__eval(draws, data)
            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo_grad(dyn, n, key, data)
