from typing import Callable, Dict

from jaxtyping import Array, Float

from bayinx.core import Flow


# TODO
class Sylvester(Flow):
    """
    A sylvester flow.

    # Attributes
    - `params`: A dictionary containing the JAX Arrays representing the flow parameters.
    - `constraints`: A dictionary of constraining transformations.
    """

    params: Dict[str, Float[Array, "..."]]
    constraints: Dict[str, Callable[[Float[Array, "..."]], Float[Array, "..."]]]
