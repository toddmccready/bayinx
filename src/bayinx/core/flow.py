from abc import abstractmethod
from typing import Callable, Dict, Tuple

import equinox as eqx
from jaxtyping import Array, Float

from bayinx.core.utils import __MyMeta


class Flow(eqx.Module, metaclass=__MyMeta):
    """
    A superclass used to define continuously parameterized diffeomorphisms for normalizing flows.

    # Attributes
    - `pars`: A dictionary of JAX Arrays representing parameters of the diffeomorphism.
    - `constraints`: A dictionary of functions that constrain their corresponding parameter.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]]

    @abstractmethod
    def forward(self, draws: Array) -> Array:
        """
        Computes the forward transformation of `draws`.
        """
        pass

    @abstractmethod
    def transform(self, draws: Array) -> Tuple[Array, Array]:
        """
        Computes the log-absolute-determinant of the Jacobian at `draws` and applies the forward transformation.

        # Returns
        A tuple of JAX Arrays containing the log-absolute-determinant of the Jacobians and transformed draws.
        """
        pass

    def __init_subclass__(cls):
        """
        Create constrain method.
        """

        def constrain(self: Flow):
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
