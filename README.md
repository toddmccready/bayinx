# <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

# TODO
- [  ] : Find some way to discern between models with all floating-point parameters and weirder models with like integer parameters. Useful for restricting variational methods like `MeanField` to `Model`s that only have floating-point parameters.
- [  ] : Look into adaptively tuning ADAM hyperparameters.
- [  ] : Figure out how to avoid the warning of setting static JAX Arrays for the normalizing flow method.
