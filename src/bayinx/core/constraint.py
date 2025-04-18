from abc import abstractmethod
from typing import Tuple

import equinox as eqx
from jaxtyping import Scalar

from bayinx.core.parameter import Parameter


class Constraint(eqx.Module):
    """
    Abstract base class for defining parameter constraints.
    """

    @abstractmethod
    def constrain(self, x: Parameter) -> Tuple[Parameter, Scalar]:
        """
        Applies the constraining transformation to a parameter and computes the log-absolute-Jacobian of the transformation.

        # Parameters
        - `x`: The unconstrained `Parameter`.

        # Returns
        A tuple containing:
            - The constrained `Parameter`.
            - A scalar Array representing the log-absolute-Jacobian of the transformation.
        """
        pass
