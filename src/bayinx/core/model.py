from abc import abstractmethod
from typing import Any, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Scalar

from bayinx.core.constraint import Constraint


class Model(eqx.Module):
    """
    An abstract base class used to define probabilistic models.

    # Attributes
    - `params`: A dictionary of JAX Arrays representing parameters of the model.
    - `constraints`: A dictionary of constraints.
    """

    params: Dict[str, Array]
    constraints: Dict[str, Constraint]

    @abstractmethod
    def eval(self, data: Any) -> Scalar:
        pass

    # Default filter specification
    def filter_spec(self):
        """
        Generates a filter specification to subset relevant parameters for the model.
        """
        # Generate empty specification
        filter_spec = jt.map(lambda _: False, self)

        # Specify JAX Array parameters
        filter_spec = eqx.tree_at(
            lambda model: model.params,
            filter_spec,
            replace=jt.map(eqx.is_array, self.params),
        )

        return filter_spec

    # Add constrain method
    @eqx.filter_jit
    def constrain_pars(self) -> Tuple[Dict[str, Array], Scalar]:
        """
        Constrain `params` to the appropriate domain.

        # Returns
        A dictionary of transformed JAX Arrays representing the constrained parameters and the adjustment to the posterior density.
        """
        t_params: Dict[str, Array] = self.params
        target: Scalar = jnp.array(0.0)

        for par, map in self.constraints.items():
            # Apply transformation
            t_params[par], ladj = map.constrain(t_params[par])

            # Adjust posterior density
            target -= ladj

        return t_params, target

    # Add default transform method
    def transform_pars(self) -> Tuple[Dict[str, Array], Scalar]:
        """
        Apply a custom transformation to `params` if needed.

        # Returns
        A dictionary of transformed JAX Arrays representing the transformed parameters.
        """
        return self.constrain_pars()
