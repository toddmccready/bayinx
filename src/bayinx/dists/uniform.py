import jax.lax as _lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real


def prob(
    x: Real[ArrayLike, "..."], lb: Real[ArrayLike, "..."], ub: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a Uniform distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `lb`:     The lower bound parameter(s).
    - `ub`:     The upper bound parameter(s).

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `lb`, and `ub`.
    """

    return 1.0 / (ub - lb) # pyright: ignore


def logprob(
    x: Real[ArrayLike, "..."], lb: Real[ArrayLike, "..."], ub: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Uniform distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `lb`:     The lower bound parameter(s).
    - `ub`:     The upper bound parameter(s).

    # Returns
    The log of the PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `lb`, and `ub`.
    """

    return _lax.log(1.0) - _lax.log(ub - lb) # pyright: ignore


def uprob(
    x: Real[ArrayLike, "..."], lb: Real[ArrayLike, "..."], ub: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Uniform distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `lb`:     The lower bound parameter(s).
    - `ub`:     The upper bound parameter(s).

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `lb`, and `ub`.
    """

    return jnp.ones(jnp.broadcast_arrays(x,lb,ub))


def ulogprob(
    x: Real[ArrayLike, "..."], lb: Real[ArrayLike, "..."], ub: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a Uniform distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `lb`:     The lower bound parameter(s).
    - `ub`:     The upper bound parameter(s).

    # Returns
    The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `lb`, and `ub`.
    """

    return jnp.zeros(jnp.broadcast_arrays(x,lb,ub))
