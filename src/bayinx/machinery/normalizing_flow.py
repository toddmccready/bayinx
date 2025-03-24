from bayinx.core import Variational
from bayinx.machinery.diffeos import Diffeomorphism


class NormalizingFlow(Variational):
    """
    An ordered collection of diffeomorphisms that map a base distribution to an approximation of a posterior distribution.

    # Attributes
    - `base`: A base distribution.
    - `diffeos`: A collection of diffeomorphisms.
    """

    base: Variational
    diffeos: list[Diffeomorphism]
