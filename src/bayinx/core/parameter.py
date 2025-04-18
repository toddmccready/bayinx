from typing import Self

import equinox as eqx
import jax.tree as jt
from jaxtyping import Array, PyTree


class Parameter(eqx.Module):
    """
    A container for a parameter of a `Model`.

    Subclasses can be constructed for custom filter specifications(`filter_spec`).

    # Attributes
    - `vals`: The parameter's value(s).
    """
    vals: Array | PyTree


    def __init__(self, values: Array | PyTree):
        # Insert parameter values
        self.vals = values

    # Default filter specification
    @property
    @eqx.filter_jit
    def filter_spec(self) -> Self:
        """
        Generates a filter specification to filter out static parameters.
        """
        # Generate empty specification
        filter_spec = jt.map(lambda _: False, self)

        # Specify Array leaves
        filter_spec = eqx.tree_at(
            lambda params: params.vals,
            filter_spec,
            replace=jt.map(eqx.is_array_like, self.vals),
        )

        return filter_spec
