from typing import Any, Callable, Self, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax as opx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Key, PyTree, Scalar
from optax import GradientTransformation, OptState, Schedule

from bayinx.core import Flow, Model, Variational


class NormalizingFlow(Variational):
    """
    An ordered collection of diffeomorphisms that map a base distribution to a normalized approximation of a posterior distribution.

    # Attributes
    - `base`: A base variational distribution.
    - `flows`: An ordered collection of continuously parameterized diffeomorphisms.
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
        flat_params, self._unflatten = ravel_pytree(params)

        self.base = base
        self.flows = flows

    @eqx.filter_jit
    def sample(self, n: int, key: Key):
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
    def eval(self, draws: Array):
        """
        Evaluate the variational density at `draws`.

        # Parameters
        - `draws`: Variational draws.

        # Returns
        A JAX Array containing the variational density for each draw.
        """
        ladj = jnp.array(0.0)

        for map in reversed(self.flows):
            # Apply inverse transformation
            draws = map.reverse(draws)

            # Evaluate adjustment
            ladj = ladj + map.inverse_ladj(draws)

        # Evaluate base density
        variational_evals = self.base.eval(draws)

        return variational_evals + ladj

    @eqx.filter_jit
    def _eval(self, draws: Array, data = None):
        variational_evals = self.base.eval(draws)

        for map in self.flows:
            # Evaluate adjustment
            variational_evals = variational_evals + map.inverse_ladj(draws)

            # Apply forward transformation
            draws = map.forward(draws)

        posterior_evals = self.eval_model(draws, data)

        return jnp.mean(posterior_evals - variational_evals)

    @eqx.filter_jit
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
    def elbo(self, n: int, key: Key, data: Any = None) -> Tuple[Scalar, PyTree]:
        """
        Estimate the ELBO and its gradient(w.r.t the variational parameters).
        """
        # Partition
        dyn, static = eqx.partition(self, self.filter_spec())

        @eqx.filter_value_and_grad
        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: Key, data: Any = None):
            # Combine
            self = eqx.combine(dyn,static)

            # Sample draws from variational distribution
            draws: Array = self.sample(n, key)

            # Evaluate ELBO
            return self._eval(draws, data)

        return elbo(dyn, n, key, data)

    @eqx.filter_jit
    def fit(
        self,
        max_iters: int,
        data: Any = None,
        learning_rate: float = 1,
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
        # Construct scheduler
        schedule: Schedule = opx.exponential_decay(
            init_value=learning_rate,
            transition_steps=max_iters,
            decay_rate=1 / max_iters,
        )

        # Initialize optimizer
        optim: GradientTransformation = opx.chain(
            opx.scale(-1.0), opx.adam(schedule, b1=0.9, b2=0.99, nesterov=True)
        )
        opt_state: OptState = optim.init(eqx.filter(self, self.filter_spec()))

        # Optimization loop helper functions
        @eqx.filter_jit
        def condition(state: Tuple[Self, OptState, Scalar, Key]):
            # Unpack iteration state
            self, opt_state, i, key = state

            return i < max_iters

        @eqx.filter_jit
        def body(state: Tuple[Self, OptState, Scalar, Key]):
            # Unpack iteration state
            self, opt_state, i, key = state

            # Update iteration
            i = i + 1

            # Update PRNG key
            key, _ = jr.split(key)

            # Compute ELBO and gradient
            _, updates = self.elbo(var_draws, key, data)

            # Compute updates
            updates, opt_state = optim.update(
                updates, opt_state, eqx.filter(self, self.filter_spec())
            )

            # Update variational distribution
            self: Self = eqx.apply_updates(self, updates)

            return self, opt_state, i, key

        # Run optimization loop
        self = lax.while_loop(
            cond_fun=condition,
            body_fun=body,
            init_val=(self, opt_state, jnp.array(0, jnp.uint32), key),
        )[0]

        return self
