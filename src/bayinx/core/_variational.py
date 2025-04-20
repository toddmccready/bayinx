from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Generic, Self, Tuple, TypeVar

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax as opx
from jaxtyping import Array, Key, PyTree, Scalar
from optax import GradientTransformation, OptState, Schedule

from ._model import Model

M = TypeVar('M', bound=Model)
class Variational(eqx.Module, Generic[M]):
    """
    An abstract base class used to define variational methods.

    # Attributes
    - `_unflatten`: A function to transform draws from the variational distribution back to a `Model`.
    - `_constraints`: The static component of a partitioned `Model` used to initialize the `Variational` object.
    """

    _unflatten: Callable[[Array], M]
    _constraints: M

    @abstractmethod
    def filter_spec(self):
        """
        Filter specification for dynamic and static components of the `Variational`.
        """
        pass

    @abstractmethod
    def sample(self, n: int, key: Key = jr.PRNGKey(0)) -> Array:
        """
        Sample from the variational distribution.
        """
        pass

    @abstractmethod
    def eval(self, draws: Array) -> Array:
        """
        Evaluate the variational distribution at `draws`.
        """
        pass

    @abstractmethod
    def elbo(self, n: int, key: Key, data: Any = None) -> Array:
        """
        Evaluate the ELBO.
        """
        pass

    @abstractmethod
    def elbo_grad(self, n: int, key: Key, data: Any = None) -> PyTree:
        """
        Evaluate the gradient of the ELBO.
        """
        pass

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0, None))
    def eval_model(self, draws: Array, data: Any = None) -> Array:
        """
        Reconstruct models from variational draws and evaluate their posterior density.

        # Parameters
        - `draws`: A set of variational draws.
        - `data`: Data used to evaluate the posterior(if needed).
        """
        # Unflatten variational draw
        model: M = self._unflatten(draws)

        # Combine with constraints
        model: M = eqx.combine(model, self._constraints)

        # Evaluate posterior density
        return model.eval(data)

    @eqx.filter_jit
    def fit(
        self,
        max_iters: int,
        data: Any = None,
        learning_rate: float = 1,
        weight_decay: float = 1e-4,
        tolerance: float = 1e-4,
        var_draws: int = 1,
        key: Key = jr.PRNGKey(0),
    ) -> Self:
        """
        Optimize the variational distribution.

        # Parameters
        - `max_iters`: Maximum number of iterations for the optimization loop.
        - `data`: Data to evaluate the posterior density with(if available).
        - `learning_rate`: Initial learning rate for optimization.
        - `tolerance`: Relative tolerance of ELBO decrease for stopping early.
        - `var_draws`: Number of variational draws to draw each iteration.
        - `key`: A PRNG key.
        """
        # Partition variational
        dyn, static = eqx.partition(self, self.filter_spec)

        # Construct scheduler
        schedule: Schedule = opx.warmup_cosine_decay_schedule(
            init_value=1e-16, peak_value=learning_rate, warmup_steps=int(max_iters/10), decay_steps=max_iters-int(max_iters/10)
        )

        # Initialize optimizer
        optim: GradientTransformation = opx.chain(
            opx.scale(-1.0), opx.nadamw(schedule, weight_decay=weight_decay)
        )
        opt_state: OptState = optim.init(dyn)

        # Optimization loop helper functions
        @eqx.filter_jit
        def condition(state: Tuple[Self, OptState, Scalar, Key]):
            # Unpack iteration state
            dyn, opt_state, i, key = state

            return i < max_iters

        @eqx.filter_jit
        def body(state: Tuple[Self, OptState, Scalar, Key]):
            # Unpack iteration state
            dyn, opt_state, i, key = state

            # Update iteration
            i = i + 1

            # Update PRNG key
            key, _ = jr.split(key)

            # Reconstruct variational
            vari = eqx.combine(dyn, static)

            # Compute gradient of the ELBO
            updates: PyTree = vari.elbo_grad(var_draws, key, data)

            # Compute updates
            updates, opt_state = optim.update(
                updates, opt_state, eqx.filter(dyn, dyn.filter_spec)
            )

            # Update variational distribution
            dyn = eqx.apply_updates(dyn, updates)

            return dyn, opt_state, i, key

        # Run optimization loop
        dyn = lax.while_loop(
            cond_fun=condition,
            body_fun=body,
            init_val=(dyn, opt_state, jnp.array(0, jnp.uint32), key),
        )[0]

        # Return optimized variational
        return eqx.combine(dyn, static)

    @eqx.filter_jit
    def posterior_predictive(
        self, func: Callable[[M], Array], n: int, data: Any = None, key: Key = jr.PRNGKey(0)
    ) -> Array:
        # Sample draws from the variational approximation
        draws: Array = self.sample(n, key)

        # Evaluate posterior predictive
        @jax.jit
        @jax.vmap
        def evaluate(draw: Array, data: Any = None):
            # Reconstruct model
            model: M = self._unflatten(draw)

            # Evaluate
            return func(model)

        return evaluate(draws)
