from typing import Any, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import optax as opx
from jaxtyping import PyTree, Scalar
from optax import GradientTransformation, OptState, Schedule

from ._model import Model


@eqx.filter_jit
def optimize_model(
    model: Model,
    max_iters: int,
    data: Any = None,
    learning_rate: float = 1,
    weight_decay: float = 0.0,
    tolerance: float = 1e-4,
) -> Model:
    """
    Optimize the dynamic parameters of the model.

    # Parameters
    - `max_iters`: Maximum number of iterations for the optimization loop.
    - `data`: Data to evaluate the model with.
    - `learning_rate`: Initial learning rate for optimization.
    - `weight_decay`: Weight decay for the optimizer.
    - `tolerance`: Relative tolerance of loss decrease for stopping early (not implemented in the loop).
    """
    # Get dynamic and static componts of model
    dyn, static = eqx.partition(model, model.filter_spec)

    # Derive gradient for posterior
    @eqx.filter_jit
    @eqx.filter_grad
    def eval_grad(dyn: Model):
        # Reconstruct model
        model: Model = eqx.combine(dyn, static)

        # Evaluate posterior
        return model.eval(data)

    # Construct scheduler
    schedule: Schedule = opx.warmup_cosine_decay_schedule(
        init_value=1e-16,
        peak_value=learning_rate,
        warmup_steps=int(max_iters / 10),
        decay_steps=max_iters - int(max_iters / 10),
    )

    optim: GradientTransformation = opx.chain(
        opx.scale(-1.0), opx.nadamw(schedule, weight_decay=weight_decay)
    )
    opt_state: OptState = optim.init(dyn)

    @eqx.filter_jit
    def condition(state: Tuple[PyTree, OptState, Scalar]):
        # Unpack iteration state
        current_opt_dyn, opt_state, i = state

        return i < max_iters

    @eqx.filter_jit
    def body(state: Tuple[PyTree, OptState, Scalar]):
        # Unpack iteration state
        dyn, opt_state, i = state

        # Update iteration
        i = i + 1

        # Evaluate gradient of posterior
        updates = eval_grad(dyn)

        # Compute updates
        updates, opt_state = optim.update(
            updates, opt_state, eqx.filter(dyn, dyn.filter_spec)
        )

        # Update model
        dyn = eqx.apply_updates(dyn, updates)

        return dyn, opt_state, i

    # Run optimization loop
    dyn = lax.while_loop(
        cond_fun=condition,
        body_fun=body,
        init_val=(dyn, opt_state, jnp.array(0, jnp.uint32)),
    )[0]

    # Return optimized model
    return eqx.combine(dyn, static)
