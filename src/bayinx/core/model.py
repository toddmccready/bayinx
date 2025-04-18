from abc import abstractmethod
from dataclasses import field, fields
from typing import Any, Self, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Scalar

from bayinx.core.constraint import Constraint
from bayinx.core.parameter import Parameter


def constrain(constraint: Constraint):
    """Defines constraint metadata."""
    return field(metadata={'constraint': constraint})


class Model(eqx.Module):
    """
    An abstract base class used to define probabilistic models.

    Annotate parameter attributes with `Parameter`.

    Include constraints by setting them equal to `constrain(Constraint)`.
    """

    @abstractmethod
    def eval(self, data: Any) -> Scalar:
        pass

    # Default filter specification
    @property
    @eqx.filter_jit
    def filter_spec(self) -> Self:
        """
        Generates a filter specification to subset relevant parameters for the model.
        """
        # Generate empty specification
        filter_spec: Self = jt.map(lambda _: False, self)

        for f in fields(self):
            # Extract attribute from field
            attr = getattr(self, f.name)

            # Check if attribute is a parameter
            if isinstance(attr, Parameter):
                # Update filter specification for parameter
                filter_spec = eqx.tree_at(
                    lambda model: getattr(model, f.name),
                    filter_spec,
                    replace=attr.filter_spec
                )

        return filter_spec


    @eqx.filter_jit
    def constrain_params(self) -> Tuple[Self, Scalar]:
        """
        Constrain parameters to the appropriate domain.

        # Returns
        A constrained `Model` object and the adjustment to the posterior.
        """
        constrained: Self = self
        target: Scalar = jnp.array(0.0)

        for f in fields(self):
            # Extract attribute
            attr = getattr(self, f.name)

            # Check if constrained parameter
            if isinstance(attr, Parameter) and 'constraint' in f.metadata:
                param = attr
                constraint = f.metadata['constraint']

                # Apply constraint
                param, laj = constraint.constrain(param)

                # Update parameters for constrained model
                constrained = eqx.tree_at(
                    lambda model: getattr(model, f.name),
                    constrained,
                    replace=param
                )

                # Adjust posterior density
                target += laj

        return constrained, target


    @eqx.filter_jit
    def transform_params(self) -> Tuple[Self, Scalar]:
        """
        Apply a custom transformation to parameters if needed(defaults to constrained parameters).

        # Returns
        A transformed `Model` object and the adjustment to the posterior.
        """
        return self.constrain_params()
