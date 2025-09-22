# From Bigram Counts to Neural Softmax: Gradients and Intuition

This document explains how we move from simple **bigram counts** to a **neural network with a softmax layer**, and why the **gradient** (and the backward pass) is the key to learning. We’ll build intuition starting from single-variable calculus and connecting step by step to the multivariable, matrix-based world of neural networks.

---

## 1. Bigram Counts → Neural Softmax

- **Bigram counts**: Count how often one character is followed by another. Store in a 27×27 matrix $N$, where $N_{ij}$ is the count of char $i$ followed by char $j$.
- **Empirical probabilities**: Convert counts to probabilities:
  $$
    P_{\text{data}}(j \mid i) = \frac{N_{ij}+1}{\sum_k (N_{ik}+1)}
  $$
- **Neural softmax model**:
  - Input = one-hot vector for current character (length 27).
  - Weight matrix $W$ of shape 27×27.
  - Logits: $ \text{logits} = x W $.
  - Softmax: 
  $$
    \hat P(j \mid i) = \frac{e^{W_{ij}}}{\sum_k e^{W_{ik}}}
  $$
- **Loss**: Negative log-likelihood (cross-entropy) between predicted probabilities and observed data.

---

## 2. Derivatives: From 1D to Many Variables

### Single-variable case
- If $y = f(x) = x$, then $dy/dx = 1$.
- Interpretation: A tiny change $\Delta x$ produces the same tiny change in $y$.

### Partial derivatives (multivariable)
- If $z = f(x,y) = x^2 + 3y$:
  - $\partial z/\partial x = 2x$
  - $\partial z/\partial y = 3$
- Interpretation: Each partial derivative is just the slope along one coordinate direction, with the others fixed.

### Gradient
- For $f(x_1, x_2, ..., x_n)$, the gradient is:
  $$
    \nabla f = \bigg( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \bigg)
  $$
- This vector points in the direction of steepest ascent.

---

## 3. Why Gradient = Sensitivity of the Loss

- For any function $L(W)$, the partial derivative $\partial L / \partial W_{ij}$ tells us:
  - **Sign**: Whether increasing that weight increases (+) or decreases (–) the loss.
  - **Magnitude**: How sensitive the loss is to small changes in that weight.
- First-order approximation (see Appendix A: Taylor Expansion for derivation):
  $$
    L(W + \Delta W) \approx L(W) + \langle \nabla L, \Delta W \rangle
  $$

  So the gradient is the local linear map of how weights affect loss.

---

## 4. Backward Pass in Neural Nets

- Forward pass:
  - Compute logits, apply softmax, compute loss.
- Backward pass:
  - Apply chain rule to compute all $\partial L / \partial W_{ij}$.
  - For softmax + cross-entropy, gradient simplifies to:
    $$
      \frac{\partial L}{\partial W_{ij}} = \hat P(j \mid i) - 1[\text{true next char}=j]
    $$
  - This shows the model adjusts weights so predicted probabilities match observed frequencies.

---

## 5. Learning Rate: Why It’s Needed

- Derivative only gives a **local** slope.
- Updates:
  $$
    W \leftarrow W - \eta \nabla_W L
  $$
  where \(\eta\) is the learning rate.
- If \(\eta\) is too small → slow progress. Too large → overshoot, diverge.
- The learning rate ensures we take steps small enough that the gradient’s “small-change” assumption is valid.

---

## 6. Intuition Recap

- **Derivative (1D)**: slope of a curve.
- **Partial derivative**: slope along one axis of a multivariable function.
- **Gradient**: collection of all slopes, pointing in steepest ascent.
- **Neural nets**: Loss depends on all weights. The backward pass computes the gradient. Updating weights opposite to the gradient (scaled by learning rate) reduces loss.
- **Bigram example**: Increasing $W_{ij}$ makes the model more likely to predict char $j$ after $i$. Gradient tells us whether that change would improve or worsen prediction accuracy.

---

## 7. Big Picture

- Gradient = the mathematical bridge from calculus to learning.
- Backprop = the algorithm that computes gradients efficiently across many weights.
- Learning rate = the dial that keeps gradient steps in the safe zone where linear approximations hold.

👉 Put together: this is how simple counts evolve into a trainable, generalizable neural language model.

---

## Appendix A. Taylor Expansion (Multivariate)  
This appendix gives the mathematical background behind the *first-order approximation* used in Section 3.

### A.1 Single-Variable Taylor Expansion
For a sufficiently smooth scalar function $f: \mathbb{R} \to \mathbb{R}$ expanded around a point $x$:
$$
f(x + h) = f(x) + f'(x)h + \tfrac{1}{2} f''(x) h^2 + \tfrac{1}{6} f^{(3)}(x) h^3 + \cdots
$$
Neglecting all but the linear term gives the 1D first-order (linear) approximation:
$$
f(x + h) \approx f(x) + f'(x) h
$$

### A.2 Multivariate Taylor Expansion
Let $F: \mathbb{R}^n \to \mathbb{R}$ be differentiable with continuous second derivatives near $\mathbf{x}$. For a small perturbation $\Delta \mathbf{x}$:
$$
F(\mathbf{x} + \Delta \mathbf{x}) = F(\mathbf{x}) + \nabla F(\mathbf{x})^{\top} \Delta \mathbf{x} + \tfrac{1}{2} \Delta \mathbf{x}^{\top} H_F(\mathbf{x}) \Delta \mathbf{x} + R_3
$$
where:
* $\nabla F(\mathbf{x})$ is the gradient (column) vector.
* $H_F(\mathbf{x})$ is the Hessian (matrix of second partial derivatives).
* $R_3$ contains third and higher order terms, typically $O(\|\Delta \mathbf{x}\|^3)$ under smoothness assumptions.

Discarding all but the linear term yields the multivariate first-order approximation:
$$
F(\mathbf{x} + \Delta \mathbf{x}) \approx F(\mathbf{x}) + \nabla F(\mathbf{x})^{\top} \Delta \mathbf{x}
$$
This is exactly the relationship used in Section 3 with $F = L$ and $\Delta \mathbf{x} = \Delta W$ (flattened weights).

### A.3 Why the Approximation Improves as Steps Shrink
The neglected quadratic term scales like $\|\Delta \mathbf{x}\|^2$. Halving the step roughly quarters its contribution. This justifies using a *learning rate* small enough that higher-order curvature does not dominate each update.

### A.4 Connection to Second-Order Methods
If we retain the quadratic term we obtain a local *quadratic model* of the loss:
$$
F(\mathbf{x} + \Delta \mathbf{x}) \approx F(\mathbf{x}) + \nabla F(\mathbf{x})^{\top} \Delta \mathbf{x} + \tfrac{1}{2} \Delta \mathbf{x}^{\top} H_F(\mathbf{x}) \Delta \mathbf{x}
$$
Minimizing this quadratic in $\Delta \mathbf{x}$ gives the classical Newton step:
$$
\Delta \mathbf{x}_{\text{Newton}} = - H_F(\mathbf{x})^{-1} \nabla F(\mathbf{x})
$$
In deep learning this is often impractical (Hessian huge / expensive), so we rely on first-order methods (SGD, Adam) that only require gradients.

### A.5 Practical Takeaways
* First-order methods implicitly assume the quadratic (and higher) terms are small per update.
* Very large gradients or learning rates break this assumption → instability.
* Techniques like learning rate schedules, gradient clipping, and normalization help keep steps in the regime where the linear model is valid.

---

