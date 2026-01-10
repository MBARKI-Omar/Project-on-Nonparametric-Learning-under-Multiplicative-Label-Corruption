## Nonparametric Learning with Instance-Dependent Label Noise

This project studies a supervised learning problem where observed labels are corrupted in an input-dependent manner. The objective is to model, estimate, and empirically analyze label noise when the corruption probability varies with the input features.

We consider a binary classification setting with observations
[
\mathcal{D}*m = {(X_i, Z_i)}*{i=1}^m,
]
where (X \in \mathbb{R}^d) is the input and (Z \in {-1,1}) is a corrupted version of an unobserved true label (Y).
The corruption follows an instance-dependent noise model: for a given input (x), the label is flipped with probability (\eta(x) \in [0, 0.5)), and kept unchanged otherwise.

This type of noise, often referred to as multiplicative or instance-dependent label corruption, is more realistic than uniform label noise and appears in many practical learning scenarios.

### Goals of the project

* Formulate a probabilistic model linking inputs, true labels, observed labels, and the noise function.
* Study identifiability issues, in particular under which assumptions the noise function can be estimated without access to true labels.
* Propose a nonparametric estimator for the noise function (\eta(x)) and provide statistical intuition for the method.
* Evaluate the estimator through simulations, analyzing the influence of sample size, dimension, and noise level.

An optional extension examines how instance-dependent label noise affects nonparametric classifiers and explores possible corrections based on the estimated noise.

### Contents

* A concise report describing the model, estimation approach, and experimental results.
* Reproducible Python code for data generation and simulations.
* Figures illustrating the behavior of the estimator under different settings.
