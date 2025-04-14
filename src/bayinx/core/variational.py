from abc import abstractmethod
from typing import Any, Callable, Self, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax as opx
from jaxtyping import Array, Float, Key, PyTree, Scalar
from optax import GradientTransformation, OptState, Schedule

from bayinx.core import Model


class Variational(eqx.Module):
    """
    A superclass used to define variational methods.

    # Attributes
    - `_unflatten`: A static function to transform draws from the variational distribution back to a `Model`.
    - `_constraints`: A static partitioned `Model` with the constraints of the `Model` used to initialize the `Variational` object.
    """

    _unflatten: Callable[[Float[Array, "..."]], Model]
    _constraints: Model

    @abstractmethod
    def sample(self, n: int, key: Key) -> Array:
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

    @abstractmethod
    def filter_spec(self):
        """
        Filter specification for dynamic and static components of the `Variational`.
        """
        pass

    def __init_subclass__(cls):
        """
        Construct methods that are shared across all VI methods.
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
            dyn, static = eqx.partition(self, self.filter_spec())

            # Construct scheduler
            schedule: Schedule = opx.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=max_iters
            )

            # Initialize optimizer
            optim: GradientTransformation = opx.chain(
                opx.scale(-1.0), opx.nadamw(schedule,weight_decay=weight_decay)
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

                # Combine variational
                vari = eqx.combine(dyn, static)

                # Compute gradient of the ELBO
                updates: PyTree = vari.elbo_grad(var_draws, key, data)

                # Compute updates
                updates, opt_state = optim.update(
                    updates, opt_state, eqx.filter(dyn, dyn.filter_spec())
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

        cls.fit = eqx.filter_jit(fit)
