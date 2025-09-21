# Maximum Likelihood Estimation (MLE)

## 1. Plain‑Language Intuition (No Math Yet)
Imagine you are:
- A coffee shop manager trying to guess what fraction of customers order espresso vs latte.
- A biologist counting how many times a certain species appears in field samples.
- A language hobbyist tallying how often one letter follows another in a list of names (our bigram example).

You observe outcomes. You write them down. You then ask:
"Which setting of my model would have made the data I actually saw the *most plausible*?"

That setting is the **Maximum Likelihood Estimate**. It is the parameter choice that says: *If the world really worked according to these parameters, the data you just handed me would be less surprising than under any other parameter choice.*

Real‑life analogy:
Think of tuning a crime story: you are a detective choosing the most plausible suspect narrative. Among all candidate explanations (suspects + motives), you pick the one that best *explains* all collected evidence. MLE does the same, but with numbers.

Another analogy (bag of marbles):
- You pull marbles (with replacement) from a bag: red, blue, blue, red, green...
- MLE says: "Set the proportion of each color in the bag to exactly the fraction you observed." That makes the observed sequence the most probable among all possible proportion assignments (under the IID-with-replacement assumption).

Key ideas:
- You never *force* the model to cover events you did **not** see; you just reward matching what you **did** see.
- If something never appeared, MLE happily assigns it probability 0 (this can later cause trouble in generation → we fix with smoothing / regularization).

## 2. Core Principle (Still Gentle)
We have:
- A model with parameters θ (theta).
- Data points $x_1, x_2, ..., x_n$ that we assume are generated (often IID) from the model distribution $p(x | \theta)$.
- The **likelihood** is the model’s joint probability of *exactly this* dataset: $L(\theta) = p(x_1, x_2, ..., x_n | \theta)$.
- The **Maximum Likelihood Estimate** is any $\hat{\theta}$ that maximizes $L(\theta)$.

Because probabilities multiply (and products can get tiny), we usually maximize the **log-likelihood** $(\ell(\theta) = log L(\theta))$. Log turns products into sums → easier, numerically stable, same maximizer (log is monotonic increasing).

## 3. Formal Definition
Given IID samples:
$$
L(\theta) = \prod_{i=1}^{n} p(x_i | \theta), \qquad \ell(\theta) = \sum_{i=1}^{n} \log p(x_i | \theta)
$$
The MLE is:
$$
\hat{\theta} = \arg\max_{\theta} \; \ell(\theta)
$$
We solve this either analytically (set derivative to zero) or numerically (gradient ascent / descent on negative log-likelihood).

## 4. Canonical Examples
### 4.1 Bernoulli (Coin Flip)
Data: $x_i \in \{0,1\}$, parameter $p = P(X=1)$.
$$
\ell(p) = \sum_{i=1}^n [x_i \log p + (1 - x_i) \log(1-p)]
$$
Differentiate, set to zero →
$$
\hat{p} = \frac{\sum_i x_i}{n} = \text{(fraction of heads)}
$$
So: “Just use the empirical frequency.”

### 4.2 Categorical (Multinomial with One Trial Each)
Outcomes $x_i$ take one of $K$ categories. Let $c_k$ = count of category $k$, $n = \sum_k c_k$. With parameter vector $\boldsymbol{\pi}$ (probabilities sum to 1):
$$
\ell(\boldsymbol{\pi}) = \sum_{k=1}^K c_k \log \pi_k \quad \text{subject to } \sum_k \pi_k = 1
$$
Using a Lagrange multiplier, solution:
$$
\hat{\pi}_k = \frac{c_k}{n}
$$
Again: empirical frequencies.

### 4.3 Gaussian (Normal) with Unknown Mean (Variance Known)
Data $x_i \sim \mathcal{N}(\mu, \sigma^2)$ ($\sigma^2$ known).
$$
\ell(\mu) = -\frac{n}{2}\log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_i (x_i - \mu)^2
$$
Maximizing is equivalent to minimizing sum of squared deviations. Result:
$$
\hat{\mu} = \frac{1}{n} \sum_i x_i
$$
Sample mean is the MLE.

### 4.4 Gaussian with Both Mean and Variance Unknown
$$
\hat{\mu} = \frac{1}{n}\sum_i x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \hat{\mu})^2
$$
(Note: denominator $n$, not $n-1$; the $n-1$ version is the unbiased *estimator*, not the MLE.)

### 4.5 Poisson (Counts)
Counts $x_i$ with mean $\lambda$:
$$
\ell(\lambda) = \sum_i [x_i \log \lambda - \lambda - \log(x_i!)] \implies \hat{\lambda} = \frac{1}{n}\sum_i x_i
$$
Again, empirical mean.

### 4.6 Bigram Language Model (Our Context)
For each adjacent character pair (current=c, next=d) you have counts $N_{cd}$. For each current character c, a conditional distribution over next characters: $P(d|c)$. The likelihood (assuming conditional independence given preceding char) factorizes row-wise; the MLE for each row is just row normalization:
$$
\hat{P}(d|c) = \frac{N_{cd}}{\sum_{d'} N_{cd'}}
$$
Exactly what we coded when we divided by the row sum.

## 5. Why Log-Likelihood? (Deeper Insight)
1. **Numerical stability**: Products of many probabilities underflow; logs convert them to manageable sums.
2. **Convexity** (sometimes): For many exponential family models the negative log-likelihood is convex → unique global optimum.
3. **Additivity**: Each data point contributes an additive term → easy for stochastic / mini-batch optimization in neural nets.

## 6. Relation to Cross-Entropy and NLL
For classification with target distribution $q$ (often a one-hot) and model distribution $p_\theta$:
$$
\text{CrossEntropy}(q, p_\theta) = - \sum_x q(x) \log p_\theta(x)
$$
When $q$ is one-hot at true class y, this reduces to $-\log p_\theta(y)$, exactly the per-example negative log-likelihood. So minimizing average cross-entropy = maximizing average log-likelihood.

## 7. General Procedure to Find an MLE
1. Specify model family $p(x|\theta)$.
2. Write likelihood or log-likelihood of observed data.
3. Differentiate w.r.t. \(\theta\).
4. Set derivative(s) = 0 (solve) and check second-order conditions (or perform gradient ascent/descent numerically if closed form is hard).
5. (If constraints) incorporate them via Lagrange multipliers or reparameterization (e.g., softmax for probabilities).
6. Validate: ensure solution is a maximum and lies in feasible region.

## 8. Edge Cases & Practical Issues
-- **Zero counts → zero probability**: Fine for pure MLE, bad for generalization (impossible events later). Fix with *smoothing* (e.g., add-one / Laplace adds a pseudo-count of 1: $\tilde{P}(d|c) = (N_{cd}+1)/(\sum_{d'} (N_{cd'}+1))$). This is *not* pure MLE; it’s a Bayesian MAP with a Dirichlet(1) prior.
-- **Overfitting**: MLE can overfit when model capacity is high (e.g., deep nets). Regularization (weight decay) = adding a prior (MAP view) or penalty.
-- **Non-identifiability**: Different $\theta$ can produce same distribution; MLE may not be unique.
- **Boundary solutions**: Probabilities collapsing to 0 or 1 at optimum (e.g., all heads). That’s still valid MLE; uncertainty not reflected unless you add a prior.

## 9. Bayesian Contrast (MAP)
MLE ignores any prior belief over parameters. Bayesian MAP (Maximum A Posteriori) instead maximizes:
$$
\log p(\theta | \text{data}) = \log p(\text{data} | \theta) + \log p(\theta) + C
$$
MAP adds $\log p(\theta)$ → a regularization term. Example: Gaussian prior on weights induces L2 penalty.

## 10. Connecting Back to the Notebook
- Raw bigram counts → MLE conditional probabilities (row normalize) → may have zeros.
- Add-one smoothing (later) modifies MLE to avoid zeros (improved sampling diversity).
- Neural model with weights W: we no longer have closed form. We compute negative log-likelihood (cross-entropy) and use gradient descent to approximate the MLE (or MAP if we add weight decay).

## 11. Worked Micro Example (Categorical)
Suppose you observed letters after start symbol '.' five times: a, a, d, a, b.
Counts: a:3, d:1, b:1 (others 0). Total =5.
MLE: $P(a|\text{'.'})=3/5, P(d|\text{'.'})=1/5, P(b|\text{'.'})=1/5, P(other)=0$.
Add-one (Laplace) smoothing if alphabet size K=27:
$$
P_{\text{Lap}}(x|.) = \frac{N_x + 1}{5 + 27}
$$

## 12. Minimal Python Illustration
```python
import torch
# pretend these are observed category indices (0..K-1)
obs = torch.tensor([0,0,2,0,1])
K = 4
counts = torch.bincount(obs, minlength=K)
MLE = counts.float() / counts.sum()
print('Counts:', counts.tolist())
print('MLE probs:', MLE.tolist())

# Negative log-likelihood of the dataset under these MLE probs
nll = -torch.log(MLE[obs]).mean()
print('Average NLL:', nll.item())

# Add-one smoothing
smoothed = (counts + 1).float() / (counts.sum() + K)
print('Laplace probs:', smoothed.tolist())
```

## 13. Summary
- MLE chooses parameters that make observed data most probable.
- For many simple distributions, MLE = "just use empirical averages / frequencies." 
- Log-likelihood simplifies math; negative log-likelihood is the loss we minimize.
- Smoothing / regularization are *not* part of pure MLE—they modify it for robustness.
- In neural networks, we rarely solve analytically: we optimize the (regularized) log-likelihood numerically with gradients.

## 14. Further Reading
- "Pattern Recognition and Machine Learning" (Bishop) – Ch. 1–2
- "Information Theory, Inference, and Learning Algorithms" (MacKay)
- Any resources on exponential family & convex optimization (for deeper theory)

---
**TL;DR:** MLE = pick the parameter setting that makes what you actually saw the least surprising. For counts, it is just normalized counts; for neural nets, it’s what gradient descent on cross-entropy is implicitly chasing.
