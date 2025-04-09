from typing import Callable, Dict

import equinox as eqx
from jaxtyping import Array


class __MyMeta(type(eqx.Module)):
    """
    Metaclass to ensure attribute types are respected.
    """

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        # Check parameters are a Dict of JAX Arrays
        if not isinstance(obj.params, Dict):
            raise ValueError(
                f"Model {cls.__name__} must initialize 'params' as a dictionary."
            )

        for key, value in obj.params.items():
            if not isinstance(value, Array):
                raise TypeError(f"Parameter '{key}' must be a JAX Array.")

        # Check constraints are a Dict of functions
        if not isinstance(obj.constraints, Dict):
            raise ValueError(
                f"Model {cls.__name__} must initialize 'constraints' as a dictionary."
            )

        for key, value in obj.constraints.items():
            if not isinstance(value, Callable):
                raise TypeError(f"Constraint for parameter '{key}' must be a function.")

        # Check that the constrain method returns a dict equivalent to `params`
        t_params: Dict[str, Array] = obj.constrain_pars()

        if not isinstance(t_params, Dict):
            raise ValueError(
                f"The 'constrain' method of {cls.__name__} must return a Dict of JAX Arrays."
            )

        for key, value in t_params.items():
            if not isinstance(value, Array):
                raise TypeError(f"Constrained parameter '{key}' must be a JAX Array.")

            if not value.shape == obj.params[key].shape:
                raise ValueError(
                    f"Constrained parameter '{key}' must have same shape as unconstrained counterpart."
                )

        # Check transform_pars

        return obj
