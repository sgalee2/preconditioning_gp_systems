# Benchmarking Preconditioners for Gaussian Process systems
Repository for studies into benchmarking system preconditioners for Gaussian Process regression models.

## Gaussian Process Regression
Gaussian Process (GP) regression assumes, given data D = (X, y), the existence of an underlying function

<p align="center">
f = y + &epsilon; 
</p>
where y|X ~ N(&mu;, K) and &epsilon; is Gaussian noise. Here K is a covariance matrix generated from the input space given a pre-defined kernel function. Training this model involves maximising the log-likelihood
<p align="center">
  L = log p(y | X, &theta;)
</p>
where &theta; is a vector of hyperparameters which define the kernel. The overhead cost of this training is the solution to a linear system in K, and the evaluation of its inverse trace and log determinant ( O(n<sup>3</sup>) ) .
<p> Predictions are made by evaluating
  <p align="center">
    &mu;<sub>*</sub> = K<sup>-1</sup>y,
  </p>
  <p align="center">
  &Sigma;<sub>*</sub> = &sigma;<sub>*</sub> - K<sub>*</sub>K<sup>-1</sup>K<sub>*</sub><sup>T</sup>
  </p>
 </p>
 
 ## Iterative Gaussian Processes
 
 Iterative GP regression avoids computing the exact solution to Kv = b, instead iteratively building the solution and terminating when we reach a certain accuracy. We do so using the *conjugate gradients* algorithm for solving positive definite systems. Fast, accurate solves can be found by first preconditioning the system with a matrix P = EE<sup>T</sup>, and instead solving the system E<sup>-1</sup>KE<sup>-T</sup>E<sup>T</sup> = E<sup>-1</sup>b. This preconditioned method uses only 2 matrix-vector products (MVPs) as the dominant overhead work and converges in k steps allowing for O(kn<sup>2</sup>) cost for GP inference which is eaily parallelised.
 
 ### Preconditioners
 
 It is standard to use an incomplete Cholesky decomposition as an out of the box preconditioner for GPs. However, we explore the posibility of using well-studied GP approximations as system preconditioners for exact GP regression.
 
 # Contact
 I can be found via email at sgalee2@liverpool.ac.uk
