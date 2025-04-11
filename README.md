# <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

The endgoal of this project is to build a Bayesian inference library that is similar in feel to `Stan`(where you can define a probabilistic model with syntax that is equivalent to how you would write it out on a chalkboard) but allows for arbitrary models(e.g., ones with discrete parameters) and offers a suite of "machinery" to fit the model; this means I want to expand upon `Stan`'s existing toolbox of methods for estimation(point optimization, variational methods, MCMC) while keeping everything performant(hence using `JAX`).

In the short-term, I'm going to focus on:
1) Implementing as much machinery as I feel is enough.
2) Figuring out how to design the `Model` superclass to have something like the `transformed pars {}` block but unifies transformations and constraints.
3) Figuring out how to design the library to automatically recognize what kind of machinery is amenable to a given probabilistic model.

In the long-term, I'm going to focus on:
1) How to get `Stan`-like declarative syntax in Python with minimal syntactic overhead(to get as close as possible to statements like `X ~ Normal(mu, 1)`), while also allowing users to work with `target` directly when needed(same as `Stan` does).
2) How to make working with the posterior as easy as possible.
    - That's a vague goal but practically it means how to easily evaluate statements like $P(\theta \in [-1, 1] | \mathcal{D}, \mathcal{M})$, or set up contrasts and evaluate $P(\mu_1 - \mu_2 > 0 | \mathcal{D}, \mathcal{M})$, or simulate the posterior predictive to generate plots, etc.

Although this is somewhat separate from the goals of the project, if this does pan out how I'm invisioning it I'd like an R formula-like syntax to shorten model construction in scenarios where the model is just a GLMM or similar(think `brms`).

Additionally, when I get around to it I'd like the package documentation to also include theoretical and implementation details for all machinery implemented(with overthinking boxes because I do like that design from McElreath's book).


# TODO
- Find some way to discern between models with all floating-point parameters and weirder models with integer parameters. Useful for restricting variational methods like `MeanField` to `Model`s that only have floating-point parameters.
- Look into adaptively tuning ADAM hyperparameters.
- Control variates for meanfield VI? Look at https://proceedings.mlr.press/v33/ranganath14.html more closely.
- Low-rank affine flow?
- https://arxiv.org/pdf/1803.05649 implement sylvester flows.
- Learn how to generate documentation lol.
- Figure out how to make transform_pars for flows such that there is no performance loss. Noticing some weird behaviour when adding constraints.
