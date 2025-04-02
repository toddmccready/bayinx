# <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

The endgoal of this project is to build a Bayesian inference library that is similar in feel to `Stan`(where you can define a probabilistic model with syntax that is equivalent to how you would write it out on a chalkboard) but allows for arbitrary models(e.g., ones with discrete parameters) and offers a suite of "machinery" to fit the model; this means I want to expand upon `Stan`'s existing toolbox of methods for estimation(point optimization, variational methods, MCMC) while keeping everything performant(hence using `JAX`).

In the short-term, I'm going to focus on getting a good desi

# TODO
- Find some way to discern between models with all floating-point parameters and weirder models with like integer parameters. Useful for restricting variational methods like `MeanField` to `Model`s that only have floating-point parameters.
- Look into adaptively tuning ADAM hyperparameters.
- Control variates for meanfield VI? Look at https://proceedings.mlr.press/v33/ranganath14.html more closely.
- Low-rank affine flow?
- Finish radial implementation.
- Constrain planar flow to always be invertible.
- https://arxiv.org/pdf/1803.05649 implement sylvester flows.
