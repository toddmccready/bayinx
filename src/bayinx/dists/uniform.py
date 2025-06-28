import jax.lax as _lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Float, Key


def prob(
    x: Float[ArrayLike, "..."], lb: Float[ArrayLike, "..."], ub: Float[ArrayLike, "..."]
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
    # Cast to Array
    x, lb, ub = jnp.asarray(x), jnp.asarray(lb), jnp.asarray(ub)

    return 1.0 / (ub - lb)


def logprob(
    x: Float[ArrayLike, "..."], lb: Float[ArrayLike, "..."], ub: Float[ArrayLike, "..."]
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
    # Cast to Array
    x, lb, ub = jnp.asarray(x), jnp.asarray(lb), jnp.asarray(ub)

    return _lax.log(1.0) - _lax.log(ub - lb)


def uprob(
    x: Float[ArrayLike, "..."], lb: Float[ArrayLike, "..."], ub: Float[ArrayLike, "..."]
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
    # Cast to Array
    x, lb, ub = jnp.asarray(x), jnp.asarray(lb), jnp.asarray(ub)

    return jnp.ones(jnp.broadcast_shapes(x.shape, lb.shape, ub.shape))


def ulogprob(
    x: Float[ArrayLike, "..."], lb: Float[ArrayLike, "..."], ub: Float[ArrayLike, "..."]
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
    # Cast to Array
    x, lb, ub = jnp.asarray(x), jnp.asarray(lb), jnp.asarray(ub)

    return jnp.zeros(jnp.broadcast_shapes(x.shape, lb.shape, ub.shape))


def sample(
    n: int,
    lb: Float[ArrayLike, "..."],
    ub: Float[ArrayLike, "..."],
    key: Key = jr.PRNGKey(0),
) -> Float[Array, "..."]:
    """
    Sample from a Uniform distribution.

    # Parameters
    - `n`: Number of draws to sample per-parameter.
    - `lb`: The lower bound parameter(s).
    - `ub`: The upper bound parameter(s).

    # Returns
    Draws from a Uniform distribution. The output will have the shape of (n,) + the broadcasted shapes of `lb` and `ub`.
    """
    # Cast to Array
    lb, ub = jnp.asarray(lb), jnp.asarray(ub)

    # Derive shape
    shape = (n,) + jnp.broadcast_shapes(lb.shape, ub.shape)

    # Construct draws
    draws = jr.uniform(key, shape, minval=lb, maxval=ub)

    return draws
