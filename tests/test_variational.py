# Imports ----
from bayinx import Model
from bayinx.dists import normal
from bayinx.machinery.meanfield import MeanField
import jax.numpy as jnp
import equinox as eqx
import pytest

# Typing ----
from typing import Dict, Callable
from jaxtyping import Array


# Tests ----
def test_meanfield():
    # Construct model
    class NormalPopulation(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]
        
        def __init__(self):
            self.params = {
                'mu': jnp.array(0.0)
            }
            
            self.constraints = {}
        
        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain()
            
            # Evaluate normal density
            return jnp.sum(normal.ulogprob(
                x = data['x'],
                mu = params['mu'],
                sigma = 1.0
            ))
    model = NormalPopulation()
    
    # Data
    data = {
        'x': jnp.array([10.0])
    }
    
    # Construct meanfield variational
    variational = MeanField(model)
    
    # Optimize variational distribution
    variational = variational.fit(10000, data = data)
    
    # Assert parameters are roughly correct
    assert abs(10.0 - variational.var_params['mean']) < 0.01
