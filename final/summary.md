---
author: Arsen Nuramatov
title: Dynamical Systems Theory in Machine Learning & Data Science
subtitle: "Final Project: Neural ODEs"
---

## Theoretical background

Neural ODEs are an architecture first introduced by Chen et al. (2018) TODO LINK
as an attempt to mitigate multiple shortcomings of older models when viewed as
ODE solvers. In this framework, architectures such as residual networks,
recurrent neural networks, and normalizing flows are viewed as Euler
discretization of a continuous transformation with uniform timestep of 1:

$$
h_{t+1} = h_t + f(h_t, \theta_t)
$$

Neural ODEs then emerge as a case with infinitesimal steps: instead of
representing evolution for a single timestep with a single hidden layer, we
induce a parameterized flow and use an off-the-shelf ODE solver:

$$
\frac{dh(t)}{dt} = f(h(t),t,\theta)
$$

This allows us to achieve several improvements:

- Memory efficiency
- Adaptive computation
- Scalable and invertible normalizing flows
- **Continuous time-series models**
  Unlike RNNs, the size of a timestep is not fixed, thus we can naturally
  incorporate data which arrives at arbitrary times
