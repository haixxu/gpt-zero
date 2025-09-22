# Where Does the "Next Char" One-Hot Go? (Target Flow & Gradient Intuition)

This note answers the specific confusion:
> Forward uses only the current character one-hot (`x_current @ W`). Where does the **next** character one-hot appear in training?

Short answer: The *target / next character* one-hot is not part of the forward matrix multiply. It appears only inside the **loss** (cross-entropy) and therefore shapes the **gradient** that updates the corresponding row of `W`.

---
## 1. Single Example Anatomy
Vocabulary size = `V`.

| Symbol | Meaning | Shape |
|--------|---------|-------|
| `i` | current character index | scalar |
| `y` | next (true) character index | scalar |
| `x` | one-hot for `i` | `(V,)` |
| `W` | weight (logit) matrix | `(V, V)` |
| `z = x @ W` | logits row for current char | `(V,)` |
| `p = softmax(z)` | predicted distribution over next chars | `(V,)` |
| `t` | one-hot for target `y` (conceptual) | `(V,)` |

Loss (negative log likelihood / cross-entropy):

$$
L = -\log p_{y} = - \sum_j t_j \log p_j
$$
Target one-hot influences training *only* via selecting `p[y]`.

---
## 2. Where the Gradient Comes From
For one example (derivative wrt logits $z$):

$$
\frac{\partial L}{\partial z_j} = p_j - t_j = p_j - \mathbf{1}[j = y]
$$

Weights:

$$
\frac{\partial L}{\partial W_{i j}} = x_i (p_j - t_j) = (p_j - t_j) \; (x_i=1), \qquad
\frac{\partial L}{\partial W_{k j}} = 0 \; (k \ne i)
$$
So only **row `i`** of `W` updates.

Interpretation: fractional reallocation of probability mass in that row toward the correct next character.

---
## 3. Counts vs. Softmax Update (Intuition)
| Aspect | Counting Model | Neural Softmax |
|--------|----------------|----------------|
| Data event `i -> y` | `N[i,y] += 1` | Logits row change: subtract lr * `(p - one_hot(y))` |
| Effect | Exact +1 to one cell | Small up for target, small down for others |
| Over many events | Frequencies accumulate | Distribution drifts toward empirical conditional |

At convergence (no regularization) logits produce probabilities equal to empirical bigram probabilities.

---
## 4. Visual Flow (Single Example)
```
current index i
    | (one-hot encode)
    v
x (1 at i)  --->  x @ W  --->  logits z (row i)  ---> softmax ---> probs p
                                                          |             \
                                                          |              (take p[y])
                                                          |                    |
                                                     target index y            |
                                                          |                    |
                                                      loss = -log p[y] <--------
                                                          |
                                                     backprop row i only
```
### Backprop — exact visual and why it only touches one row

Forward (single example)
```
i (index) 
    ↓ one-hot
x = one_hot(i)         # shape (V,)
    ↓ matmul
z = x @ W              # shape (V,)   <-- this is exactly row W[i,:]
    ↓ softmax
p = softmax(z)         # shape (V,)
    ↓ pick target
loss = -log p[y]
```

Backward (grad flow)
```
loss
    ↓ d/dz  (through softmax+log)
∂L/∂z = p - t          # vector of length V, where t = one_hot(y)
    ↓ d/dW  (through z = x @ W)
∂L/∂W = x[:,None] * (p - t)[None,:]   # outer product
```

Why that means only one row updates
- x is a one-hot with 1 at position i ⇒ x[k]=0 for k≠i and x[i]=1.
- Therefore ∂L/∂W[k,:] = x[k] * (p - t) = 0 for all k ≠ i.
- Only ∂L/∂W[i,:] = (p - t) is nonzero — the gradient is an entire vector that updates the looked-up row.

Sign intuition (why target increases, others decrease)
- For the target coordinate y: (p_y - 1) < 0 (unless p_y=1) → gradient descent step W[i,y] -= lr*(p_y-1) increases W[i,y].
- For other j: (p_j - 0) = p_j > 0 → W[i,j] decreases by lr * p_j.
- Net effect: reallocate probability mass inside row i toward y.

Compact summary (update rule used in SGD)

$$
W_{i,:} \leftarrow W_{i,:} - \eta (p - t)
$$
Batched view: sum these per-example (or average). Grouping by current index i yields

$$
\frac{\partial L}{\partial W_{i,:}} \propto N_i (\hat p_i - \tilde p_i)
$$
where $N_i$ = number of training examples with current index $i$, $\hat p_i$ = model distribution for that row, $\tilde p_i$ = empirical distribution.
which is why, at optima, predicted_dist_i → empirical bigram distribution for that row.


---
## 5. Micro Numeric Example
Data: from `.` we saw `.a, .a, .b` → empirical probabilities `[a:2/3, b:1/3]`.

Initial (random) logits row for `.`:
```
[ 0.20, -0.10, 0.05 ]  -> softmax ≈ [0.384, 0.285, 0.331]
```
One training example with target `a`:
```
probs - one_hot(a) = [0.384, -0.715, 0.331]
Update (SGD):

$$
W_{\.,:} \leftarrow W_{\.,:} - \eta (p - t)
$$
Target logit increases (its gradient component is negative); others decrease slightly.
```
After many such steps (2/3 times target=a, 1/3 times target=b):
```
Logits approach something like [-10, log 2, log 1]  (any constant shift ok)
Softmax -> [~0, 2/3, 1/3]
```
Matches counts.

---
## 6. Batched View = Aggregated Fractional Counts
Group all examples having current $i$:

$$
\frac{\partial L}{\partial W_{i,*}} \propto N_i (\hat p_i - \tilde p_i)
$$
Setting gradient ≈ 0 ⇒ predicted == empirical.

---
## 7. Code Illustration (Explicit Target One-Hot)
```python
import torch, torch.nn.functional as F
V = 5
# Fake single example
i = torch.tensor(2)      # current
y = torch.tensor(4)      # next
W = torch.randn(V, V, requires_grad=True)

x = F.one_hot(i, num_classes=V).float()   # (V,)
logits = x @ W            # (V,) row i
probs = logits.softmax(dim=0)

# Explicit one-hot target (usually we just use probs[y])
t = F.one_hot(y, num_classes=V).float()
loss = -(t * probs.log()).sum()  # same as -log(probs[y])
loss.backward()

print('Row grads (only row i should be nonzero):')
print(W.grad.abs().sum(dim=1))
```
Expected output: only index `i` row has non-zero gradient.

---
## 8. Analogy: Question Card vs Answer Card
- Input one-hot = **question card** (“Which row should I read?”)
- Row of `W` = **guesses distribution** (scores for all answers)
- Target index = **answer card** (“Which one should have been highest?”)
- Loss/gradient = **teacher correction** (nudge guess distribution toward answer)

---
## 9. FAQ
**Q: Why not multiply target one-hot in forward?**  
A: Because you need all candidate next-character probabilities for sampling / loss. The target is only used to pick which log-prob to penalize.

**Q: Why exp?**  
To turn unconstrained real logits into positive pseudo-counts so normalization mirrors `counts / row_sum` semantics.

**Q: Why only one row updates?**  
Input one-hot has exactly one 1 → gradient paths touch only that row.

---
## 10. Mantra
"The target one-hot supervises *which element* of the looked-up row should receive the highest probability; it doesn’t participate in the lookup—only in the correction."

---
## 11. Appendix: Softmax Derivative (Why $\partial L/\partial z = p - t$)

We rely on the identity

$$
\frac{\partial L}{\partial z} = p - t
$$

for cross-entropy loss with logits $z$, probabilities $p = \operatorname{softmax}(z)$, and one-hot target $t$. Here is the full derivation and intuition.

### 11.1 Softmax Definition
For logits vector $z \in \mathbb{R}^V$:

$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

Let $S = \sum_k e^{z_k}$ so $p_j = e^{z_j} / S$.

### 11.2 Partial Derivative of p_j wrt z_l
Two cases:
1. l = j
$$
\frac{\partial p_j}{\partial z_j} = \frac{e^{z_j} S - e^{z_j} e^{z_j}}{S^2} = \frac{e^{z_j}(S - e^{z_j})}{S^2} = \frac{e^{z_j}}{S}\left(1 - \frac{e^{z_j}}{S}\right) = p_j (1 - p_j)
$$
2. l ≠ j
$$
\frac{\partial p_j}{\partial z_l} = - \frac{e^{z_j} e^{z_l}}{S^2} = - \frac{e^{z_j}}{S} \frac{e^{z_l}}{S} = - p_j p_l, \quad l \ne j
$$
Combine both with a Kronecker delta $\delta_{jl}$:

$$
\frac{\partial p_j}{\partial z_l} = p_j (\delta_{jl} - p_l)
$$
This is the classic **softmax Jacobian** entry.

### 11.3 Cross-Entropy Loss
Cross-entropy with a one-hot target $t$ (index $y$):

$$
L = - \sum_j t_j \log p_j = - \log p_y
$$

Its derivative w.r.t. probabilities:

$$
\frac{\partial L}{\partial p_j} = - \frac{t_j}{p_j}
$$
(Only the target j=y contributes nonzero.)

### 11.4 Chain Rule to Logits
$$
\frac{\partial L}{\partial z_l} = \sum_j \left(-\frac{t_j}{p_j}\right) p_j (\delta_{jl} - p_l) = \sum_j (-t_j)(\delta_{jl} - p_l)
$$

Because $t$ is one-hot at $y$:

$$
\frac{\partial L}{\partial z_l} = - (\delta_{yl} - p_l) = p_l - t_l
$$
Done.

### 11.5 Vector / Matrix Form
The Jacobian matrix of softmax alone is

$$
J_{\text{softmax}}(z) = \operatorname{diag}(p) - p p^T
$$
Cross-entropy with one-hot target multiplies by `(-t / p)` elementwise then sums, collapsing the matrix multiply to the simple vector `p - t`.

### 11.6 Why It Simplifies Nicely
- The negative log picks only one probability component (target) in the forward pass.
- The softmax Jacobian structure makes all off-diagonal interactions aggregate into `-p_l`.
- The one-hot sparsity eliminates the sum over j except at the target.
Result: highly efficient backward pass (no need to form Jacobian explicitly).

### 11.7 Intuition (Probability Conservation)
Gradient components sum to zero:

$$
\sum_l (p_l - t_l) = \Big(\sum_l p_l\Big) - \Big(\sum_l t_l\Big) = 1 - 1 = 0
$$
So updates redistribute mass within the row rather than creating/destroying total probability — matching the “reallocate marbles” picture.

### 11.8 Quick Sanity Code
```python
import torch, torch.nn.functional as F
z = torch.randn(5, requires_grad=True)
y = torch.tensor(2)
p = z.softmax(dim=0)
loss = -p.log()[y]
loss.backward()
print('autograd:', z.grad)
with torch.no_grad():
    manual = p - F.one_hot(y, num_classes=5).float()
print('manual  :', manual)
```
They match.

### 11.9 Checklist
| Concept | Can Explain? |
|---------|--------------|
| Softmax Jacobian entry formula | Yes |
| Why Jacobian never built explicitly | Yes |
| Why gradient sums to zero | Yes |
| How cross-entropy + softmax collapse to p - t | Yes |

### 11.10 One-Line Memory Hook
"Softmax + cross-entropy gradient = predicted minus target."  (Same pattern appears in logistic regression, multi-class classification, etc.)

---
*Use this file as a focused supplement when the role of the target one-hot feels unclear.*
