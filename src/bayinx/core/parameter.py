from typing import Generic, Self, TypeVar

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

T = TypeVar('T', bound=PyTree)
class Parameter(eqx.Module, Generic[T]):
    """
    A container for a parameter of a `Model`.

    Subclasses can be constructed for custom filter specifications(`filter_spec`).

    # Attributes
    - `vals`: The parameter's value(s).
    """
    vals: T


    def __init__(self, values: T):
        # Insert parameter values
        self.vals = values

    def __call__(self) -> T:
        return self.vals

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
