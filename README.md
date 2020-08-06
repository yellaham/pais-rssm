# Particle Adaptive Importance Sampling for Regime-Switching State-Space Models

## Summary
This repository contains an implementation of a particle adaptive importance sampling (PAIS) method for the 
Bayesian analysis of regime-switching state-space models. The software implementation can be utilized to study general 
Bayesian models - all that is required is that the user supplies a method to compute unbiased estimates of the 
likelihood function. This particular software implementation focuses on regime-switching state-space models. The 
algorithm is applied to analyze the population dynamics of Adélie penguins through both synthetic data and real data 
collected from a public database. 


## Examples
The code related to each of the following examples is under the "pais-rssm/examples/" directory. 

### Example 0: Multivariate Gaussian Target Distribution
In the first example, we approximate a multivariate Gaussian distribution using the implemented AIS sampler in this
repository. The following figure shows the approximated target distribution obtained from running the sampler:

![alt text](https://github.com/yellaham/pais-rssm/blob/master/figures/ex0_target_contour_plot.png "KDE Plot (Ex. 0)")


### Example 1: Switching Linear-Gaussian State-Space Model
In this example, our goal is to estimate the hidden states in a linear-Gaussian state-space model, where the parameters
of the state-space model can arbitrarily switch from time instant to the next. The following figure shows tracking of 
the hidden state under model uncertainity:

![alt text](https://github.com/yellaham/pais-rssm/blob/master/figures/ex1_tracking_performance.png "Tracking Performance (Ex. 1)")

We can also plot the model selection performance to see how well the algorithm does in detecting which model is 
represented at each time instant. 

![alt text](https://github.com/yellaham/pais-rssm/blob/master/figures/ex1_model_selection_performance.png "Model Detection (Ex. 1)")
