# Imports ----
import jax
import equinox as eqx
import optax as opx
import jax.lax as lax, jax.numpy as jnp, jax.random as jr
from jax.flatten_util import ravel_pytree
from bayinx.dists import normal
from functools import partial

# Typing ----
from typing import Dict, Callable, Tuple, Any, Self
from jaxtyping import Array, Key, Scalar
from optax import OptState
from bayinx import Model


class MeanField(eqx.Module):
    var_params: Dict[str, Array]
    __unflatten: Callable[[Array], Model] = eqx.field(static = True)
    __constraints: Model = eqx.field(static = True)
    
    def __init__(self, model: Model):
        # Partition model
        params, self.__constraints = eqx.partition(model, eqx.is_array)
        
        # Flatten params component
        flat_params, self.__unflatten = ravel_pytree(params)
        
        # Initialize variational parameters
        self.var_params = {
            'mean': flat_params,
            'log_std': jnp.zeros(flat_params.size, dtype = flat_params.dtype)
        }
    
    
    @eqx.filter_jit
    def sample(self, n: int, key: Key) -> Array:
        """
        Sample from the flattened variational distribution `n` times.
        """
        
        # Sample flattened parameters
        flat_params: Array = jr.normal(
            key = key, 
            shape = (n, self.var_params['mean'].size)
        ) * jnp.exp(self.var_params['log_std']) + self.var_params['mean']
        
        return flat_params
    
    
    @eqx.filter_jit
    def eval_variational(self, samples: Array) -> Array:
        """
        Evaluate the variational density at `samples`.
        """
        return normal.ulogprob(
            x = samples, 
            mu = self.var_params['mean'],
            sigma = jnp.exp(self.var_params['log_std'])
        ).sum(axis = 1)
    
    
    @partial(jax.vmap, in_axes = (None, 0, None))
    @eqx.filter_jit
    def unflatten_and_eval_model(self, sample: Array, data: Any = None) -> Array:
        """
        Unflatten variational samples and evaluate their posterior density.
        """
        # Unflatten variational sample
        model: Model = self.__unflatten(sample)
        
        # Combine with constraints
        model: Model = eqx.combine(model, self.__constraints)
        
        # Evaluate
        return model.eval(data)
    
    
    @eqx.filter_value_and_grad
    @eqx.filter_jit
    def elbo(self, n: int, key: Key, data: Any = None) -> Scalar:
        """
        Estimate the ELBO and its gradient(w.r.t the variational parameters).
        """
        # Sample flattened variational distribution
        param_samples: Array = self.sample(n, key)
        
        # Evaluate posterior at samples
        posterior_evals: Array = self.unflatten_and_eval_model(param_samples, data)
        
        # Evaluate variational distribution
        variational_evals: Array = self.eval_variational(param_samples)
        
        # Evaluate ELBO
        return jnp.mean(posterior_evals - variational_evals)
    
    
    @eqx.filter_jit
    def fit(self, 
            max_iters: int, 
            key: Key = jr.PRNGKey(0),
            learning_rate: float = 1, 
            tolerance: float = 1e-4,
            var_samples: int = 1, 
            data: Any = None) -> Self:
        """
        Optimize the variational distribution.
        """
        # Construct scheduler
        schedule = opx.exponential_decay(
            init_value = learning_rate,
            transition_steps = max_iters,
            decay_rate = 1 / max_iters
        )
        
        # Initialize optimizer
        optim: opx.GradientTransformation = opx.chain(
            opx.scale(-1.0), 
            opx.adam(schedule, b1 = 0.5, b2 = 0.5, nesterov = True)
        )
        opt_state = optim.init(eqx.filter(self, eqx.is_array))
        
        @eqx.filter_jit
        def condition(state: Tuple[MeanField, OptState, Scalar, Key]):
            # Unpack iteration state
            self, opt_state, i, key = state
            
            return i < max_iters
        
        @eqx.filter_jit
        def body(state: Tuple[MeanField, OptState, Scalar, Key]):
            # Unpack iteration state
            self, opt_state, i, key = state
            
            # Update iteration
            i = i + 1
            
            # Update PRNG key
            key, _ = jr.split(key)
            
            # Calculate ELBO and gradient
            _, updates = self.elbo(
                n = var_samples,
                key = key,
                data = data
            )
            
            # Calculate updates
            updates, opt_state = optim.update(
                updates, opt_state, eqx.filter(self, eqx.is_array)
            )
            
            # Update variational distribution
            self = eqx.apply_updates(self, updates)
            
            return self, opt_state, i, key
        
        # Run optimization loop
        self = lax.while_loop(
            cond_fun = condition,
            body_fun = body,
            init_val = (self, opt_state, jnp.array(0, jnp.uint32), key)
        )[0]
        
        return self
