# Bayinx: <ins>Bay</ins>esian <ins>In</ins>ference with JA<ins>X</ins>

The original aim of this project was to build a PPL in Python that is similar in feel to Stan or Nimble(where there is a nice declarative syntax for defining the model) and allows for arbitrary models(e.g., ones with discrete parameters that may not be just integers); most of this goal has been moved to [baycian](https://github.com/toddmccready/baycian) for the foreseeable future.

Part of the reason for this move is that Rust's ability to embed a "nice" DSL is comparitively easier due to [Rust macros](https://doc.rust-lang.org/rust-by-example/macros/dsl.html); I can define syntax similar to Stan and parse it to valid Rust code. Additionally, the current state of bayinx is relatively functional(plus/minus a few things to clean-up and documentation) and it offers enough functionality for one of my other projects: [disize](https://github.com/toddmccready/disize)! I plan to rewrite disize in Python with JAX, and bayinx makes it easy to handle constraining transformations, filtering for parameters for gradient calculations, etc.

Instead, this project is narrowing on implementing much of Stan's functionality(restricted to continuously parameterized models, point estimation + vi + mcmc, etc) without most of the nice syntax, at least for versions `0.4.#`. Therefore, people will work with `target` directly and return the density like below:

```py
class NormalDist(Model):
    x: Parameter[Array] = define(shape = (2,))

    def eval(self, data: Dict[str, Array]):
        # Constrain parameters
        self, target = self.constrain_params() # this does nothing for the current model

        # Evaluate x ~ Normal(10.0, 1.0)
        target += normal.logprob(self.x(), 10.0, 1.0).sum()

        return target
```

I have ideas for using a context manager and implementing `Node`: `Observed`/`Stochastic` classes that will try and replicate what `baycian` is trying to do, but that is for the future and versions `0.4.#` will retain the functionality needed for disize.


# TODO
- For optimization and variational methods offer a way for users to have custom stopping conditions(perhaps stop if a single parameter has converged, etc).
- Control variates for meanfield VI? Look at https://proceedings.mlr.press/v33/ranganath14.html more closely.
- Low-rank affine flow?
- https://arxiv.org/pdf/1803.05649 implement sylvester flows.
- Learn how to generate documentation.
- Figure out how to make transform_pars for flows such that there is no performance loss. Noticing some weird behaviour when adding constraints.
- Look into adaptively tuning ADAM hyperparameters for VI.
