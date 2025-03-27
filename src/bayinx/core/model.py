from abc import abstractmethod
from typing import Any, Callable, Dict

import equinox as eqx
from jaxtyping import Array, Scalar

from bayinx.core.utils import __MyMeta


class Model(eqx.Module, metaclass=__MyMeta):
    """
    A probabilistic model.

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
        """
        Create constrain method.
        """

        def constrain(self: Model) -> Dict[str, Array]:
            """
            Constrain `params` to the appropriate domain.

            # Returns
            A dictionary of transformed JAX Arrays representing the constrained parameters.
            """
            t_params = self.params

            for par, map in self.constraints.items():
                t_params[par] = map(t_params[par])

            return t_params

        cls.constrain = eqx.filter_jit(constrain)
