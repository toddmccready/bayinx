# MARK: Imports ----
import jax.lax as _lax

## Typing
from jaxtyping import Real, UInt, Array, ArrayLike


# MARK: Functions ----
def prob(x: UInt[ArrayLike, "..."],
         p: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
    """
    The probability mass function (PMF) for a Bernoulli distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `p`:     The probability parameter(s).

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    
    
    return _lax.pow(p, x) * _lax.pow(1 - p, 1 - x)

def logprob(x: UInt[ArrayLike, "..."],
            p: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
    """
    The log probability mass function (log PMF) for a Bernoulli distribution.
    
    # Parameters
    - `x`:  Value(s) at which to evaluate the log PMF.
    - `p`:  The probability parameter(s).
    
    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    
    
    return x * _lax.log(p) + (1 - x) * _lax.log(1 - p)