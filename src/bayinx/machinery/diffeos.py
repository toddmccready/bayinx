# Imports ----

import equinox as eqx

# Typing ----


class Diffeomorphism(eqx.Module):
    forward: list
