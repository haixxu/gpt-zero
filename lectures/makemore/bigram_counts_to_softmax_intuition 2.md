# From Explicit Bigram Counts to a Neural Softmax Layer: A Unifying Intuition

This document bridges the *counting-based* bigram language model (frequency table → normalized probabilities) and the *neural* bigram model (one-hot → linear layer → softmax). It explains why they are mathematically equivalent at optimum, how gradients mimic incremental counting, and why the neural view generalizes beyond simple tables.

---
## 1. The Count Matrix `N`
`N` is a 27×27 tally: rows = current character, columns = next character. Each row `i` defines an empirical categorical distribution after normalization:
$$P_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}} = p(\text{next}=j \mid \text{current}=i).$$
This is the maximum likelihood estimate (MLE) for each row independently.

---
## 2. Neural Parameter Matrix `W` = Learnable Log-Counts
With one-hot inputs, a linear layer `logits = x @ W` just selects the row of `W` corresponding to the active (current) character. So row `i` of `W` holds 27 *scores* for the next character—analogous to a row of `N`, but continuous and trainable.

If we *set* `W = log(N + 1)` (Laplace smoothing), then a softmax over rows exactly reproduces smoothed empirical probabilities.

---
## 3. Softmax as Differentiable Row Normalization
Given a row of scores (logits) `w_{i,:}`, softmax produces:
$$p_{i,j} = \frac{e^{w_{i,j}}}{\sum_k e^{w_{i,k}}}.$$
Compare to normalized counts:
$$P_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}}.$$
If `w_{i,j} = log N_{i,j}` (or `log(N_{i,j}+1)`), the two match *exactly*. Exponentiation gives pseudo-counts; dividing by their sum normalizes.

---
## 4. Row Shift Invariance and Log Space
Softmax is invariant to adding any constant to all elements in a row:
$$\text{softmax}(w_{i,:}) = \text{softmax}(w_{i,:} + c_i \mathbf{1}).$$
Counts have a similar property: scaling a row of `N` by a constant leaves normalized probabilities unchanged. Moving to log space converts that multiplicative freedom into additive freedom.

---
## 5. Deriving the Optimum (Why the Model Recovers Counts)
Negative log-likelihood for one row `i`:
$$\mathcal{L}_i = -\sum_j N_{i,j} \log p_{i,j}, \quad p_{i,j} = \frac{e^{w_{i,j}}}{\sum_k e^{w_{i,k}}}.$$
Gradient:
$$\frac{\partial \mathcal{L}_i}{\partial w_{i,j}} = -N_{i,j} + (\sum_k N_{i,k}) p_{i,j}. $$
Setting to zero gives:
$$p_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}}.$$
Thus any logits of the form `w_{i,j} = log N_{i,j} + c_i` (row constant) minimizes the loss for that row. The neural model **learns** the empirical distribution (with smoothing if you bias it so) when capacity matches the table.

---
## 6. What a Single Gradient Step "Feels" Like
For one training bigram `(current=i, next=j)`:
- Probability of correct `j` too low → increase its logit.
- All other logits in row slightly decrease (since softmax probabilities must still sum to 1).
This mirrors incrementing `N_{i,j}` and renormalizing: a differentiable analogue of counting.

---
## 7. Tiny 3-Symbol Example
Symbols: `.` (0), `a` (1), `b` (2). Suppose after `.` we observed: `a, a, b` → counts row = `[0, 2, 1]`.
- Empirical probs: `[0/3, 2/3, 1/3]`.
- Add-one smoothing: counts become `[1,3,2]` → probs `[1/6, 3/6, 2/6]`.
- Set logits to `log([1,3,2]) ≈ [0, 1.0986, 0.6931]` → softmax reproduces `[1/6, 3/6, 2/6]`.

---
## 8. Why Not Just Divide Raw Linear Outputs?
We need: positivity + normalization + smooth gradients. Softmax provides all three via `exp` (positive) and division (normalization), and yields stable analytical gradients. Raw linear outputs can be negative and unbounded, making direct normalization invalid.

---
## 9. Cross-Entropy and KL View
Loss:
$$\text{NLL} = -\frac{1}{T}\sum_t \log p_{t,y_t}$$
is the empirical cross-entropy between the one-hot true distribution and model distribution. Minimizing it = minimizing KL(empirical || model). With full capacity (one row per context), optimum equals the empirical conditional distribution.

---
## 10. Why Learn If Counting Already Solves It?
Counting is a special case (fixed bigram context). Learning:
- Adds regularization (e.g., weight decay → smoother distributions).
- Enables initialization from priors (uniform, smoothed counts, etc.).
- Scales to richer contexts (trigrams, neural embeddings, Transformers).
- Shares statistical strength via parameterization (hidden layers, embeddings) rather than keeping rows totally independent.

---
## 11. Analogy
Picture 27 boards (contexts). Each has 27 notes (next chars). Counting = increment integer scribbles. Neural training = start with random scores; each mistake slightly raises the correct note and lowers others. After many passes, relative heights match empirical frequencies. Softmax = "treat heights as logs of pseudo-counts and normalize".

---
## 12. Numerically Stable Softmax (Preview)
For large magnitude logits you subtract the row max:
$$p_{i,j} = \frac{e^{w_{i,j} - m_i}}{\sum_k e^{w_{i,k} - m_i}}, \quad m_i = \max_k w_{i,k}.$$
Row shift invariance guarantees identical probabilities, avoids overflow.

---
## 13. Programmatic Equivalence Check
```python
# Given integer count matrix N (27x27)
import torch

# Laplace smoothing + log
W_init = (N + 1).float().log()  # pretend each bigram seen once

# Softmax via logits
logits = W_init
counts_like = logits.exp()
probs_from_W = counts_like / counts_like.sum(1, keepdims=True)

# Direct smoothed probabilities
direct = (N + 1).float()
direct /= direct.sum(1, keepdims=True)

assert torch.allclose(probs_from_W, direct)
print("Softmax(log(N+1)) matches Laplace-smoothed probabilities exactly.")
```

---
## 14. Gradient = Adjusted Counting (Formula)
For a single example with true next index `y` in row `i`:
- `∂NLL/∂w_{i,y} = p_{i,y} - 1`
- `∂NLL/∂w_{i,k} = p_{i,k}` for `k ≠ y`
SGD step subtracts this: increases the correct logit (since `p_{i,y} - 1 < 0` if `p_{i,y} < 1`) and decreases others proportional to their probability mass.

---
## 15. Mental Compression
| Concept | Counts Model | Neural Model |
|---------|--------------|--------------|
| Parameters | Integers in `N` | Real logits in `W` |
| Smoothing | Add-one to counts | Add constant bias / prior logits |
| Normalize | Divide by row sums | Softmax row-wise |
| Update Mechanism | Increment counts | Gradient adjusts logits |
| Optimum | Empirical conditionals | Same distributions |
| Extension | Hard for longer context | Add layers / embeddings |

**Keep in head:** *A softmax layer over one-hot inputs is a learnable, smoothed, real-valued generalization of a conditional count table.*

---
## 16. Where Power Emerges
Once you: (a) replace one-hot with learned embeddings, (b) feed multiple previous chars via nonlinear layers (MLP, RNN, Transformer), the same softmax head now models complex context-dependent distributions. The bigram counting intuition still anchors the mechanics: unnormalized scores → exponentiate → normalize → sample / evaluate.

---
## 17. Optional Next Steps
- Initialize `W` from `log(N+1)` and compare training speed.
- Replace one-hot with an embedding matrix and verify equivalence for dimension = 27.
- Extend to trigrams: context becomes a pair of indices; counting explodes (27² rows) but neural nets share parameters.

---
## 18. Quick FAQ
**Q: Why call them *logits* instead of log-counts?** Historical ML terminology; “logits” = pre-softmax scores, not guaranteed to be literal logs of anything—though here they *can be interpreted* as log-counts.

**Q: Do we lose anything by moving from exact counts to learned logits?** No for bigrams; we gain differentiability and extensibility.

**Q: Why does adding a constant to a row not matter?** Because both numerator and denominator in softmax get multiplied by `e^{c}` and cancel.

**Q: What enforces probabilities sum to 1?** The row-wise division in softmax; always true by construction.

---
## 19. Summary Diagram
```
counts path: N  --(+1)-->  N+1  --row normalize-->  P  --sample--> text
neural path: W  --exp-->   exp(W)  --row normalize--> softmax(W) --> text
                 ^ learns to mimic (smoothed) empirical distribution
```

---
If any single step above still feels opaque, note which section and we can zoom in further with micro-examples or derivative printouts.
