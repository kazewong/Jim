# Jim's parameter transform system

!!! warning
    **Heavy development**: This is a work in progress and is subject to change. If you have any questions, please feel free to open an issue.

The parameterization of some fundamental quantities can have significant impacts on the performance of the sampler. For example, the masses can be parameterized in the component mass spaces $\mathcal{M}_1$-$\mathcal{M}_2$ or chirp mass-mass ratio $\mathcal{M}_c$-$q$. Because of the strong correlation in the component mass space, the geometry is much harder for a sampler to explore. This can lead to slow convergence and poor performance. The chirp mass-mass ratio space is much more isotropic and easier to explore, which is often the choice to expose to the sampler.

On the other hand, defining a prior in the component mass space is much more intuitive. The common prior of choice is uniform in the component mass space with some maximum and minimum mass. One may want to define the prior in the component mass space then sample in the chirp mass space. To make the problem even worse, there is yet another set of paraemeters one may want to choose, which is the set of parameters the model may want to take. For example, the waveform generator in ripple takes the symmetric mass ratio $\eta$ as input instead of the mass ratio.

So in a general setting, there could be three sets of parametrizations we can choose for our problem: a parameterization $\vec{\theta_{prior}}$ which we want to define our prior in, a parameterization $\vec{z}$ which we want the sampler to see, and a parameterization $\vec{\theta}_{model}$ which the model takes. To facilitate the transformation between these parameterizations, we introduce a naming system and a transform system to handle this.

A sketch of the transform system is shown below:
![A sketch of the transform system](prior_system_diagram.png)

Fundamentally, all `jim` needs is to pass a reasonable posterior function `posterior(z) = likelihood(t_1(z)) + prior(t_2(z))` to `flowMC`, so `flowMC` can sample the function. Here, `likelihood` and `prior` are functions that one pass in a vector of numbers (and data), and it returns a scalar. `z` is the vector of variables the sampler sees, and `t_1` and `t_2` are the transforms that transform the variables between the sampler and the model. 

Since we want to sample in `z`, ultimately we want to have a prior defined in `z`, meaning we can evaluate `p_Z(z)`. However, this is could be non-trivial, such as defining a prior in the $\mathcal{M}_c$-$q$ space that is equivalent to a uniform prior in the $\mathcal{M}_1$-$\mathcal{M}_2$ space requires a very complex expression. Instead of directly writing down the prior distribution in `z`, we can start by defining the prior in a space that is more intuitive, then transform the prior to the space `z` using a transform. 

By carrying the Jacobian of the transform, we can define the prior in the space `z` by transforming the prior in the intuitive space, $p_Z(z) = p_X(x) \left| \frac{dx}{dz} \right|$, where $x = t_2(z)$. The tricky bit here is the user often wants to start from the prior $x$ and define the transform **to** the sample space `z`, but here the transform is defined **from** the sample space `z` to the prior space $x$. So in order to make the prior-to-sample transform work in practice, the transform has to be **bijective**, which implies they are invertible and have a well-defined determinant of the Jacobian.

On the other hand, the likelihood is often related to a model instead of coming from a distribution, meaning as long as one can transform the parameters to feed into the model, there is no limitation on the transform.. So the transform for the likelihood is strictly from the sample space `z` to the model space $y = t_1(z)$. For this reason, the transform for the likelihood does not need to be bijective, it can even by a N-to-M transform.

# Prior

## Primatives


## Setting up Priors
Prior represents the prior knowledge on the probability density distribution of the event parameters $\theta$. In Jim, we could set up priors using the prior class `jimgw.prior`. Suppose we want to define a uniform prior distribution for the parameter $x$, we could call the `UniformPrior` class:

```
from jimgw.prior import UniformPrior
prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
```

Jim provides a number of built-in prior classes, including:
- `UniformPrior`
- `SinePrior`
- `CosinePrior`
- `PowerLawPrior`

A complete list of prior classes available can be found in the Jim documentation (not available yet).

### Multi-dimensional Priors
When working with multi-dimensional parameter space, we will usually want to define priors for each individual parameter first:

```
from jimgw.prior import UniformPrior
prior_x = UniformPrior(0.0, 1.0, parameter_names=["x"])
prior_y = UniformPrior(0.0, 2.0, parameter_names=["y"])
```

Once we have the individual priors defined, we can call the class `CombinePrior` to combine them into one single prior object:

```
from jimgw.prior import CombinePrior
prior = CombinePrior(
    [
        prior_x,
        prior_y,
    ]
)
```

The `CombinePrior` object takes a list of `jimgw.prior` objects as argument and combines them into one single prior object that can be fed to the Jim sampler as input.

# Transforms
In general, there could be three sets of parametrizations we need to work with in our problem:

1. A parameterization $\vec{\theta_{prior}}$ which we want to define our prior in
2. a parameterization $\vec{z}$ which we want the sampler to see 
3. a parameterization $\vec{\theta_{model}}$ which the model takes

To handle these different parameterizations, Jim provides a naming system and a transform system to facilitate the necessary transformations between the different parameterizations.

## Setting up Sample Transformations
![Sample Transformations](sample_transform.png)
As mentioned earlier, we may have different parameterizations for defining the priors ($\theta_{prior}$) and for the sampler to see ($z$). Therefore, we have to set up sample transformations that can convert between these two parameterizations.

### Bound-to-unbound Transformations
In general, we want the sampler to sample in an unconstrained parameter space, in which the set of parameters $z$ are well-defined over the entire real space. This prevents the samples from crossing into regions of the parameter space where the value is not well-defined.

However, the parameterization that we use to define the priors, $\theta_{prior}$, often contain parameters with hard bounds. Therefore, we need a bound-to-unbound transformation to map the bounded parameters into an unbounded representation. In a nutshell, bound-to-unbound transformation in Jim utilize the logit function to map real numbers from $(0, 1)$ to real numbers in $(-\infty, +\infty)$. The logit function is given by:

$$
logit(\theta) = \ln{\frac{\theta}{1-\theta}}
$$

To set up bound-to-unbound transformation, we call the transformation class `BoundToUnbound` provided in `jimgw.transforms`:

```
from jimgw.transforms import BoundToUnbound
sample_transform = [
    BoundToUnbound(name_mapping = [["x"], ["x_unbounded"]], original_lower_bound=0.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["y"], ["y_unbounded"]], original_lower_bound=0.0, original_upper_bound=10.0),
]
```

These sample transformations are now contained in the list `sample_transform`, which will be passed to the `Jim` sampler. When the transformations are executed in `Jim`, they will be applied in the order as they appear in the list.

### Other Sample Transformations
Other than bound-to-unbound transformations `BoundToUnbound`, Jim also provides transformations in `jimgw.single_event.transforms` for compact binary coalescence (CBC) parameter estimation problems. In these problems, we may want to sample on different parameters than the ones we used to define the priors. For example, if we have defined prior on component masses $m_1$ and $m_2$, but wish to sample on chirp mass $M_c$ and mass ratio $q$ instead. We could contain the function `ComponentMassesToChirpMassMassRatioTransform` in the sample transform list:

```
from jimgw.single_event.transforms import ComponentMassesToChirpMassMassRatioTransform
sample_transforms = [
    ComponentMassesToChirpMassMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "q"]]),
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=10.0, original_upper_bound=80.0),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=0.125, original_upper_bound=1.0),
]
```

Again, we should always include a `boundToUnbound` transform at the end of the list of `sample_transform`.


## Setting up Likelihood Transformations
![Likelihood transformation](likelihood_transform.png)
Again, the parameterization $\vec{\theta_{model}}$ used by the model may be different from the parameterization $\vec{\theta_{prior}}$ used to define the prior. Therefore, we need to set up likelihood transformations that map from the parameterization $\vec{\theta_{model}}$ to the parameterization $\vec{\theta_{prior}}$.

Similar to the sample transformations, we need to define a list of transformations that will map from $\vec{\theta_{prior}}$ space to $\vec{\theta_{model}}$ space.

```
likelihood_transforms = [
    ComponentMassesToChirpMassMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "q"]]),
    MassRatioToSymmetricMassRatioTransform(name_mapping=[["q"], ["eta"]]),
]
```

Again when the transformations are executed in `Jim`, they will be applied in the order as they appear in the list.
