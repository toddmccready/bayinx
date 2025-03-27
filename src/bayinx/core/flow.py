from abc import abstractmethod
from typing import Callable, Dict

import equinox as eqx
from jaxtyping import Array, Float

from bayinx.core.utils import __MyMeta


class Flow(eqx.Module, metaclass=__MyMeta):
    """
    A continuously parameterized diffeomorphism.

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
    def reverse(self, draws: Array) -> Array:
        """
        Computes the reverse transformation of `draws`.
        """
        pass

    @abstractmethod
    def inverse_ladj(self, draws: Array) -> Array:
        """
        Computes the log-absolute-determinant of the Jacobian at `draws` for the reverse transformation.

        # Returns
        A JAX Array containing the log-absolute-determinant of the Jacobians.
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
