import jax.lax as _lax
from jaxtyping import Array, ArrayLike, Float, Real

__PI = 3.141592653589793


def prob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return _lax.exp(-0.5 * _lax.square((x - mu) / sigma)) / ( # pyright: ignore
        sigma * _lax.sqrt(2.0 * __PI)
    )


def logprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log PDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The log of the PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return -_lax.log(sigma * _lax.sqrt(2.0 * __PI)) - 0.5 * _lax.square((x - mu) / sigma) # pyright: ignore


def uprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the uPDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return _lax.exp(-0.5 * _lax.square((x - mu) / sigma)) / sigma # pyright: ignore


def ulogprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log uPDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return -_lax.log(sigma) - 0.5 * _lax.square((x - mu) / sigma) # pyright: ignore
