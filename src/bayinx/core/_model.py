from abc import abstractmethod
from dataclasses import field, fields
from typing import Any, Dict, Optional, Self, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar

from ._constraint import Constraint
from ._parameter import Parameter


def define(
    shape: Optional[Tuple[int, ...]] = None,
    init: Optional[PyTree] = None,
    constraint: Optional[Constraint] = None
):
    """Define a parameter."""
    metadata: Dict = {}
    if constraint is not None:
            metadata["constraint"] = constraint

    if isinstance(shape, Tuple):
        metadata["shape"] = shape
    elif isinstance(init, PyTree):
        metadata["init"] = init
    else:
        raise TypeError("Neither 'shape' nor 'init' were given as proper arguments.")

    return field(metadata = metadata)


class Model(eqx.Module):
    """
    An abstract base class used to define probabilistic models.

    Annotate parameter attributes with `Parameter`.

    Include constraints by setting them equal to `define(Constraint)`.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)

        # Auto-initialize parameters based on `define` metadata
        for f in fields(cls):
            if "shape" in f.metadata:
                # Construct jax Array with correct dimensions
                setattr(obj, f.name, Parameter(jnp.zeros(f.metadata["shape"])))
            elif "init" in f.metadata:
                # Slot in given 'init' object
                setattr(obj, f.name, Parameter(f.metadata["init"]))
            else:
                raise RuntimeError("neither 'shape' or 'init' found in field metadata.")

        return obj

    def __init__(self):
        return self

    @abstractmethod
    def eval(self, data: Any) -> Scalar:
        pass

    # Default filter specification
    @property
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
            if isinstance(attr, Parameter) and ("constraint" in f.metadata):
                param = attr
                constraint = f.metadata["constraint"]

                # Apply constraint
                param, laj = constraint.constrain(param)

                # Update parameters for constrained model at same node
                constrained = eqx.tree_at(
                    lambda model: getattr(model, f.name), constrained, replace=param
                )

                # Adjust posterior density
                target += laj

        return constrained, target

    def transform_params(self) -> Tuple[Self, Scalar]:
        """
        Apply a custom transformation to parameters if needed(defaults to constrained parameters).

        # Returns
        A transformed `Model` object and the adjustment to the posterior.
        """
        return self.constrain_params()
