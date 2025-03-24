# Imports ----
import equinox as eqx

# Typing ----
from typing import Dict, Callable, Any
from jaxtyping import Array, Scalar
from abc import abstractmethod


class __ModelMeta(type(eqx.Module)):
    # Metaclass to ensure Model attribute types are respected.
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        
        # Check parameters are a Dict of jax Arrays
        if not isinstance(obj.params, Dict):
            raise ValueError(f"Model {cls.__name__} must initialize 'params' as a Dict.")
        
        for key, value in obj.params.items():
            if not isinstance(value, Array):
                raise TypeError(f"Parameter '{key}' must be a jax Array.")
        
        
        # Check constraints are a Dict of functions
        if not isinstance(obj.constraints, Dict):
            raise ValueError(f"Model {cls.__name__} must initialize 'constraints' as a Dict.")
        
        for key, value in obj.constraints.items():
            if not isinstance(value, Callable):
                raise TypeError(f"Constraint for parameter '{key}' must be a function.")
        
        
        # Check that the constrain method returns a dict equivalent to `params`
        t_params: Dict[str, Array] = obj.constrain()
        
        if not isinstance(t_params, Dict):
            raise ValueError(f"The 'constrain' method of {cls.__name__} must return a Dict of jax Arrays.")
        
        for key, value in t_params.items():
            if not isinstance(value, Array):
                raise TypeError(f"Constrained parameter '{key}' must be a jax Array.")
            
            if not value.shape == obj.params[key].shape:
                raise ValueError(f"Constrained parameter '{key}' must have same shape as unconstrained counterpart.")
        
        return obj

class Model(eqx.Module, metaclass = __ModelMeta):
    """
    A probabilistic model.
    """
    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    
    @abstractmethod
    def eval(self, data: Any) -> Scalar:
        pass
    
    def __init_subclass__(cls):
        """
        Create constrain method.
        """
        
        # Construct constrain method
        def constrain(self: Model):
            t_params = self.params
            
            for par, map in self.constraints.items():
                t_params[par] = map(t_params[par])
            
            return t_params
        cls.constrain = eqx.filter_jit(constrain)
