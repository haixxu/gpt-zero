# Bigram Counts Foundations

A gentle, concrete introduction to building a **character-level bigram language model** *purely by counting* before touching neural networks or gradients. This is the “feel the data” phase: you literally tally how often one character follows another, turn those tallies into probabilities, and sample new names.

> Goal: Start from raw text (a list of names) and end with a function that can generate new, name-like strings using only counting + simple probability.

```text
OVERVIEW PIPELINE (COUNTS PHASE)
--------------------------------
Raw names
   | tokenize chars + pad '.'
   v
Enumerate bigrams (ch1->ch2)
   | accumulate integer frequencies
   v
Count matrix N  (V x V)
   | (+1 smoothing optional)
   v
Row normalize -> Probability matrix P
   | sequential sampling using rows
   v
Generated name strings
```

---
## 1. What Is a Bigram?
A **bigram** is an ordered pair of consecutive symbols. For characters:
- In the word `anna`, the padded form (with start/end markers `.`) is: `. a n n a .`
- Its bigrams: `.a`, `an`, `nn`, `na`, `a.`

```text
PADDING & BIGRAM EXTRACTION EXAMPLE
Word:        a   n   n   a
Pad:      .  a   n   n   a  .
Bigrams: (.a)(an)(nn)(na)(a.)
          ^  ^   ^   ^   ^
Each arrow is a training event: current -> next
```

Each bigram answers: “What character came next after this one?”

We model the conditional probability:
```
P(next = c2 | current = c1)
```
for every possible pair of characters.

---
## 2. Why Add Start/End Tokens?
We add a special token `.` to both start and end:
- Start `.` captures how names begin (what letters tend to appear first)
- End `.` captures how names terminate (what letters tend to appear last)

Without them you can’t generate *where* to begin or *when* to stop.

```text
ROLE OF '.' TOKEN
Start '.'  -> first letter distribution  (row 0)
Any letter -> '.' end token probability  (col 0)
Generation loop stops when '.' sampled again.
```

---
## 3. Load and Inspect the Data
```python
words = open('names.txt').read().splitlines()
print(len(words), 'names, sample:', words[:5])
```
Ask simple questions (minimum length, maximum length) to build intuition.

---
## 4. First Counting Pass (Dictionary)
Use a Python dictionary keyed by `(ch1, ch2)` tuples.
```python
from collections import defaultdict
bigram_counts = defaultdict(int)

for w in words:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        bigram_counts[(ch1, ch2)] += 1

# Peek at a few most frequent
top = sorted(bigram_counts.items(), key=lambda kv: -kv[1])[:20]
for (a,b), c in top:
    print(f'{a}{b}: {c}')
```
This gives raw *frequency*, not probability.

Interpretation: If `('a','n')` count is high, many names contain `an`.

```text
INTERNAL VIEW (DICT)
{('.','a'): 523, ('a','n'): 410, ('n','n'): 95, ... }
Key = tuple (current, next)
Value = integer frequency
```

---
## 5. From Counts to Probabilities (Per-Current-Character)
We need, for each current character `a`, the distribution over possible next characters `b`:
```
P(b | a) = count(a,b) / sum_{b'} count(a,b')
```

A helper to gather conditional distributions:
```python
from collections import defaultdict
next_totals = defaultdict(int)
for (ch1, ch2), cnt in bigram_counts.items():
    next_totals[ch1] += cnt

# probability for a particular pair
def prob(ch1, ch2):
    return bigram_counts[(ch1, ch2)] / next_totals[ch1]

print('P(n | a) =', prob('a','n'))
```

Problem: Some pairs were *never* seen → probability 0. That breaks log-likelihood math & sampling diversity.

```text
UNSEEN BIGRAM ISSUE
If ('q','z') never appears -> count = 0
Row normalize -> probability exactly 0
Any future name needing q->z transition => impossible event.
Solution: smoothing (pretend each pair seen ≥1 time)
```

---
## 6. The Vocabulary and Index Mapping
Moving to tensors later is easier with integer IDs.
```python
# Extract unique characters and build mappings
chars = sorted(list({c for w in words for c in w}))
chars = ['.'] + chars          # ensure '.' is index 0
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
V = len(chars)
print('Vocab size:', V, chars)
```

```text
MAPPINGS
Character -> Index (stoi)
Index     -> Character (itos)
Indices enable: array addressing, tensor operations, later neural embeddings
```

---
## 7. Dense Count Matrix N (Shape V x V)
Rows = current char, Columns = next char.
```python
import torch
N = torch.zeros((V, V), dtype=torch.int32)
for w in words:
    chars_in = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars_in, chars_in[1:]):
        i = stoi[ch1]
        j = stoi[ch2]
        N[i, j] += 1

print('Row for start token (.) raw counts:\n', N[0])
```
Why a matrix?
- Constant-time row slicing
- Vectorized math (normalization, smoothing, sampling prep)
- Mirrors how a later neural weight matrix will look

```text
STRUCTURE OF N (conceptual)
          NEXT (columns)
        .  a  b  c  ...
CUR .  [*  *  *  *  ...]
R   a  [*  *  *  *  ...]
O   b  [*  *  *  *  ...]
W   c  [*  *  *  *  ...]
S  ... [...           ]
Each row sums to total occurrences of that current char.
```

---
## 8. Visualizing (Optional but Insightful)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.imshow(N, cmap='Blues')
plt.title('Raw Bigram Count Matrix')
plt.xlabel('Next char index')
plt.ylabel('Current char index')
plt.show()
```
Dark rows/columns highlight frequent transitions.

```text
HEATMAP INTERPRETATION
Darker cell (i,j) => more frequent transition current=i -> next=j
Row 0 ('.') pattern => starting letter distribution
Column 0 ('.') pattern => which letters often terminate names
```

---
## 9. Row Normalization: Turning Counts into Probabilities
```python
P = N.float()
row_sums = P.sum(1, keepdim=True)  # shape (V,1)
P = P / row_sums                   # broadcasting division
# Each row now sums (approximately) to 1
print('Check row 0 sum:', P[0].sum())
```
Now `P[i, j] ≈ P(next=j | current=i)` *except* for rows with zeros (if a character never appeared, row sum was 0 ⇒ division by 0). In this dataset all characters appear, but some transitions are 0 → probability 0.

```text
ROW NORMALIZATION
Row i counts:  c0  c1  c2  ...  c26
Row sum S = c0 + c1 + ... + c26
Divide each by S => probabilities summing to 1
```

---
## 10. Add-One (Laplace) Smoothing
Avoid zeroes by pretending each bigram was seen once.
```python
Ps = (N + 1).float()
Ps /= Ps.sum(1, keepdim=True)
print('Minimum prob after smoothing:', Ps[Ps>0].min().item())
```
Interpretation: We reallocate a tiny bit of mass from frequent transitions to unseen ones → better robustness when generating or evaluating new data.

```text
SMOOTHING EFFECT (single row example)
Raw counts:       [5, 0, 12, 1]
+1 smoothing -->  [6, 1, 13, 2]
Raw probs:        [5/18, 0/18, 12/18, 1/18]
Smoothed probs:   [6/22, 1/22, 13/22, 2/22]
Zero replaced by small positive mass; frequent events slightly reduced.
```

---
## 11. Sampling New Names from the Smoothed Model
We perform **ancestral sampling**: start at `'.'`, repeatedly pick a next character from the row distribution until `'.'` is chosen again.
```python
g = torch.Generator().manual_seed(42)  # reproducibility

for _ in range(5):
    out = []
    ix = 0   # index for '.' start
    while True:
        p = Ps[ix]                      # 1D tensor of length V
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        ch = itos[ix]
        if ch == '.':
            break
        out.append(ch)
    print(''.join(out))
```
What happens inside:
1. Look up row (current state distribution)
2. Sample next index
3. Append char if not end token
4. Repeat until end

Because probabilities are independent of history beyond the current character, this is a **Markov chain of order 1**.

```text
SAMPLING LOOP (STATE MACHINE VIEW)
state = '.'
while True:
  probs = Ps[state]
  next_state ~ Categorical(probs)
  if next_state == '.': stop
  emit(next_state)
  state = next_state
```

---
## 12. Evaluating a Name’s Log-Likelihood
Compute product of conditional probabilities (sum of logs) over its bigrams:
```python
import math

def log_likelihood(name, Pmat=Ps):
    chars_in = ['.'] + list(name) + ['.']
    total = 0.0
    for ch1, ch2 in zip(chars_in, chars_in[1:]):
        i, j = stoi[ch1], stoi[ch2]
        prob = Pmat[i, j].item()
        total += math.log(prob)
    return total

print('log p(andrej) =', log_likelihood('andrej'))
```
Lower (more negative) log-likelihood indicates the model considered the sequence less typical.

```text
LOG-LIKELIHOOD EXAMPLE (name = 'ana')
P(.->a) * P(a->n) * P(n->a) * P(a->.)
log = sum of individual log probabilities
Higher (less negative) => more plausible under model
```

---
## 13. Sanity Checks & Pitfalls
| Issue | Symptom | Fix |
|-------|---------|-----|
| Forgot start/end tokens | Generated names never stop | Pad with `.` at both ends |
| Division by zero | NaN rows | Ensure every char appears at least once; or add smoothing first |
| Sampling bias | Strange repetition | Use `torch.multinomial(p, 1)` not `torch.randint` |
| Not seeding | Non-reproducible demos | Use `torch.Generator().manual_seed(seed)` |
| Zero probabilities | `-inf` log-likelihood | Apply Laplace smoothing |

```text
DEBUG TIP
If sampling collapses (repeats same short pattern), print the row probabilities for that pattern's final char; it may dominate due to skewed counts.
```

---
## 14. Why This Matters Before Neural Nets
| Concept Here | Neural Net Analogue |
|--------------|---------------------|
| Matrix `N` (counts) | Weight matrix `W` (logits) |
| Row normalization | Softmax over logits |
| Add-one smoothing | Logit priors / initialization |
| Sampling from row | Sampling from softmax distribution |
| Log-likelihood sum | Cross-entropy loss (negative mean log prob) |

By mastering counting, the *neural* version becomes: “replace fixed integers with learnable real numbers and optimize them so implied probabilities better match data.”

```text
COUNTS -> NEURAL MAPPING
N (int)   replaces  W (float)
Row norm  replaces  softmax
Add-one   replaces  initialization / bias prior
Increment replaces  gradient update
```

---
## 15. Minimal End-to-End Script (All Together)
```python
import torch, math, matplotlib.pyplot as plt
words = open('names.txt').read().splitlines()
chars = ['.'] + sorted(list({c for w in words for c in w}))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
V = len(chars)

# Count matrix
N = torch.zeros((V,V), dtype=torch.int32)
for w in words:
    cs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(cs, cs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Smoothed probabilities
P = (N + 1).float()
P /= P.sum(1, keepdim=True)

# Sample
g = torch.Generator().manual_seed(0)
for _ in range(5):
    out = []
    ix = 0
    while True:
        ix = torch.multinomial(P[ix], 1, generator=g).item()
        if ix == 0: break
        out.append(itos[ix])
    print(''.join(out))
```
You can run this in a clean environment and it will produce plausible name-like outputs.

```text
SCRIPT DATA FLOW
load -> build vocab -> fill N -> +1 -> normalize -> sample loop
```

---
## 16. Where to Go Next
1. Replace raw counts with a learnable matrix `W` (already conceptually the same shape).  
2. Introduce a differentiable mapping: one-hot -> linear layer -> softmax.  
3. Optimize `W` via gradient descent to maximize log-likelihood (minimize negative log-likelihood).  
4. Expand context (trigrams, n-grams) or move to embeddings + MLP/Transformer.

---
## 17. Key Takeaways
- Counting bigrams = building empirical conditional distributions.
- Laplace smoothing ensures every transition remains *possible*.
- Sampling uses those conditional distributions sequentially to grow strings.
- The entire neural softmax layer is a smooth, trainable generalization of this exact pipeline.

**Keep this picture in mind:**
```
Raw text -> tally transitions -> count matrix N -> ( +1 ) -> normalize rows -> P -> sample/log-likelihood
```
Neural model just makes `N` *learnable* as real numbers and updates them with gradients instead of integer increments.

```text
END-TO-END ANALOGY
+---------+    +----------+    +-------------+    +--------------+
| Raw Txt | -> |  Bigrams | -> | Count Table | -> | Probabilities | -> sample
+---------+    +----------+    +-------------+    +--------------+
                           (classical counts)

Later neural model:
One-hot idx -> Linear W -> Softmax -> Probabilities
```

---
*Feel free to request a companion version that interleaves this with the neural softmax explanation or adds visuals for top transitions.*
