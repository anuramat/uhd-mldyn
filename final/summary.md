---
author: Arsen Nuramatov
title: Dynamical Systems Theory in Machine Learning & Data Science
subtitle: "Final Project: Neural ODEs"
---

## Theoretical background [^1]

Neural ODEs are an architecture introduced as an attempt to mitigate multiple
shortcomings of older models such as residual networks, recurrent neural
networks, and normalizing flows. In the proposed framework they are viewed as
Euler discretization of a continuous transformation with a uniform timestep:

$$
h_{t+1} - h_t = f(h_t, \theta_t)
$$

Neural ODEs then emerge as a limiting case with infinitesimal steps: instead of
using separate hidden layer for each step of time evolution, we define a
parameterized flow using a neural network and then use an off-the-shelf ODE
solver:

$$
\frac{dh(t)}{dt} = f(h(t),t,\theta)
$$

The main obstacle in training neural ODEs is performing backpropagation through
the ODE solver, as naive differentiation through the operations of forward pass
is problematic due to high memory cost and numerical errors. Instead, the
*adjoint sensitivity method* is used, which is linear in problem size, has low
memory cost, and explicitly controls numerical error.

The resulting approach has several benefits:

- Memory efficiency: not storing intermediate quantities of the forward pass
  allows us to train our models with constant memory cost as a function of depth
- Adaptive computation: modern ODE solvers allow us to adjust the balance
  between approximation accuracy and computational complexity both during
  training and inference
- Scalable and invertible normalizing flows: since the resulting mapping is a
  diffeomorphism, ...
- Continuous time-series models: unlike RNNs, the size of a timestep is not
  fixed, thus we can naturally incorporate data which arrives at arbitrary times

[^1]: ["Neural Ordinary Differential Equations", RTQ Chen et al.](https://doi.org/10.48550/arXiv.1806.07366)
