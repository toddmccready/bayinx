from abc import abstractmethod
from typing import Any, Callable, Dict

import equinox as eqx
from jaxtyping import Array, Scalar

from bayinx.core.utils import __MyMeta


class Model(eqx.Module, metaclass=__MyMeta):
    """
    A superclass used to define probabilistic models.

    # Attributes
    - `params`: A dictionary of JAX Arrays representing parameters of the model.
    - `constraints`: A dictionary of functions that constrain their corresponding parameter.
    """

    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]

    @abstractmethod
    def eval(self, data: Any) -> Scalar:
        pass

    def __init_subclass__(cls):
        # Add constrain method
        def constrain_pars(self: Model) -> Dict[str, Array]:
            """
            Constrain `params` to the appropriate domain.

            # Returns
            A dictionary of transformed JAX Arrays representing the constrained parameters.
            """
            t_params = self.params

            for par, map in self.constraints.items():
                t_params[par] = map(t_params[par])

            return t_params

        cls.constrain_pars = eqx.filter_jit(constrain_pars)

        # Add transform_pars method if not present
        if not callable(getattr(cls, "transform_pars", None)):
            def transform_pars(self: Model) -> Dict[str, Array]:
                """
                Apply a custom transformation to `params` if needed.

                # Returns
                A dictionary of transformed JAX Arrays representing the transformed parameters.
                """
                return self.constrain_pars()

            cls.transform_pars = eqx.filter_jit(transform_pars)
