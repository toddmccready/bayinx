from abc import abstractmethod
from typing import Callable, Dict, Self, Tuple

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
    def adjust_density(self, draws: Array) -> Tuple[Array, Array]:
        """
        Computes the log-absolute-determinant of the Jacobian at `draws` and applies the forward transformation.

        # Returns
        A tuple of JAX Arrays containing the log-absolute-determinant of the Jacobians and transformed draws.
        """
        pass

    def __init_subclass__(cls):
        # Add contrain_pars method
        def constrain_pars(self: Self):
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
            def transform_pars(self: Self) -> Dict[str, Array]:
                """
                Apply a custom transformation to `params` if needed.

                # Returns
                A dictionary of transformed JAX Arrays representing the transformed parameters.
                """
                return self.constrain_pars()

            cls.transform_pars = eqx.filter_jit(transform_pars)
