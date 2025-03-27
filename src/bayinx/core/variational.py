from abc import abstractmethod
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, Scalar

from bayinx.core import Model


class Variational(eqx.Module):
    """
    A superclass used to define VI methods.

    # Attributes
    - `_unflatten`: A static function to transform draws from the variational distribution back to `Model`.
    - `_constraints`: A static partitioned `Model` object with the constraints of the original `Model` subclass used to initialize the `Variational` object.
    """

    _unflatten: Callable[[Float[Array, "..."]], Model]
    _constraints: Model

    @abstractmethod
    def sample(self, n: int, key: Key) -> Array:
        """
        Sample from the variational distribution `n` times.
        """
        pass

    @abstractmethod
    def eval(self, draws: Array) -> Array:
        """
        Evaluate the variational distribution at `draws`.
        """
        pass

    def __init_subclass__(cls):
        """
        Create more methods.
        """

        def eval_model(self, draws: Array, data: Any = None) -> Array:
            """
            Reconstruct models from variational draws and evaluate their posterior density.

            # Parameters
            - `draws`: A set of variational draws.
            - `data`: Data used to evaluate the posterior(if needed).
            """
            # Unflatten variational draw
            model: Model = self._unflatten(draws)

            # Combine with constraints
            model: Model = eqx.combine(model, self._constraints)

            # Evaluate posterior density
            return model.eval(data)

        cls.eval_model = jax.vmap(eqx.filter_jit(eval_model), (None, 0, None))

        def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
            """
            Estimate the ELBO and its gradient(w.r.t the variational parameters).
            """

            # Sample draws from variational distribution
            draws: Array = self.sample(n, key)

            # Evaluate posterior density for each draw
            posterior_evals: Array = self.eval_model(draws, data)

            # Evaluate variational density for each draw
            variational_evals: Array = self.eval(draws)

            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        cls.elbo = eqx.filter_value_and_grad(eqx.filter_jit(elbo))
