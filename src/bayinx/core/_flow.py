from abc import abstractmethod
from typing import Callable, Dict, Self, Tuple

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, Float


class Flow(eqx.Module):
    """
    An abstract base class for a flow(of a normalizing flow).

    # Attributes
    - `params`: A dictionary of JAX Arrays representing parameters of the diffeomorphism.
    - `constraints`: A dictionary of simple functions that constrain their corresponding parameter.
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
        Computes the log-absolute-Jacobian at `draws` and applies the forward transformation.

        # Returns
            A tuple of JAX Arrays containing the transformed draws and log-absolute-Jacobians.
        """
        pass

    # Default filter specification
    @property
    def filter_spec(self):
        """
        Generates a filter specification to subset relevant parameters for the flow.
        """
        # Generate empty specification
        filter_spec = jtu.tree_map(lambda _: False, self)

        # Specify JAX Array parameters
        filter_spec = eqx.tree_at(
            lambda flow: flow.params,
            filter_spec,
            replace=jtu.tree_map(eqx.is_array, self.params),
        )

        return filter_spec

    def constrain_params(self: Self):
        """
        Constrain `params` to the appropriate domain.

        # Returns
        A dictionary of transformed JAX Arrays representing the constrained parameters.
        """
        t_params = self.params

        for par, map in self.constraints.items():
            t_params[par] = map(t_params[par])

        return t_params

    def transform_params(self: Self) -> Dict[str, Array]:
        """
        Apply a custom transformation to `params` if needed.

        # Returns
        A dictionary of transformed JAX Arrays representing the transformed parameters.
        """
        return self.constrain_params()
