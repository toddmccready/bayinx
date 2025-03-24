import jax.lax as lax
from jaxtyping import Array, ArrayLike, Real, UInt


# MARK: Functions ----
def prob(x: UInt[ArrayLike, "..."], p: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
    """
    The probability mass function (PMF) for a Bernoulli distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `p`:     The probability parameter(s).

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """

    return lax.pow(p, x) * lax.pow(1 - p, 1 - x)


def logprob(x: UInt[ArrayLike, "..."], p: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
    """
    The log probability mass function (log PMF) for a Bernoulli distribution.

    # Parameters
    - `x`:  Value(s) at which to evaluate the log PMF.
    - `p`:  The probability parameter(s).

    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """

    return lax.log(p) * x + (1 - x) * lax.log(1 - p)
