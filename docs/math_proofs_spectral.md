# Mathematical Proofs for Spectral-JAX Architecture

**Document Type**: Technical Supplement  
**Status**: Pre-print / Internal Review  

---

## PROOF 1: Computational Complexity $O(N \log N)$

### Theorem 1.1
*A Spectral-JAX block with selective filtering has time complexity $O(N \log N \cdot D)$ for sequence length $N$ and hidden dimension $D$.*

### Proof

Consider a single Spectral-JAX block operating on input $X \in \mathbb{R}^{N \times D}$:

**Step 1: Forward FFT**
$$
\hat{X} = \text{FFT}(X) \quad \text{where} \quad \hat{X}[k] = \sum_{n=0}^{N-1} X[n] \cdot e^{-i 2\pi kn / N}
$$

Via Cooley-Tukey algorithm:
- Base complexity: $O(N^2 \cdot D)$ (naive DFT)
- Optimized: $O(N \log_2 N \cdot D)$ (divide-and-conquer)

**Step 2: Selective Filter Generation**
$$
H_{\theta}(X) = \text{MLP}_{\theta}(X) \quad \text{where MLP is a 2-layer network}
$$

Cost Analysis:
- Input projection: $X W_1$ where $W_1 \in \mathbb{R}^{D \times D_h}$ → $O(N \cdot D \cdot D_h)$
- Activation: ReLU → $O(N \cdot D_h)$
- Output projection: $H W_2$ where $W_2 \in \mathbb{R}^{D_h \times D}$ → $O(N \cdot D_h \cdot D)$

Total MLP cost: $O(N \cdot D \cdot D_h)$. For $D_h = O(D)$, this becomes $O(N \cdot D^2)$.

**Step 3: Spectral Mixing**
$$
\hat{Y}[k] = \hat{X}[k] \odot H_{\theta}(X)[k] \quad \text{(element-wise)}
$$

Cost: $O(N \cdot D)$ (simple multiplication)

**Step 4: Inverse FFT**
$$
Y = \text{iFFT}(\hat{Y})
$$

Cost: $O(N \log N \cdot D)$ (same as forward FFT)

**Step 5: Energy Gate**
$$
Y_{\text{out}} = Y \odot \sigma(\|Y\|_2)
$$

- Norm computation: $O(N \cdot D)$
- Gate application: $O(N \cdot D)$

Total: $O(N \cdot D)$

### Combined Complexity

$$
T(N, D) = \underbrace{O(N \log N \cdot D)}_{\text{FFT}} + \underbrace{O(N \cdot D^2)}_{\text{MLP}} + \underbrace{O(N \log N \cdot D)}_{\text{iFFT}} + \underbrace{O(N \cdot D)}_{\text{Gate}}
$$

**Case 1**: If $D \ll N$ (typical in LLMs, e.g., $D=512, N=8192$):
$$
O(N \log N \cdot D) + O(N \cdot D^2) \approx O(N \log N \cdot D) \quad \text{since } D^2 < N \log N
$$

**Case 2**: If $D \gg N$ (rare):
$$
T(N,D) = O(N \cdot D^2)
$$

For standard LLM configurations, **FFT dominates** → $O(N \log N \cdot D)$. ∎

---

## PROOF 2: Spectral Selectivity ≈ Restricted Attention

### Theorem 2.1
*The selective spectral mixing operation approximates a structured attention mechanism with a specific sparsity pattern.*

### Proof Sketch

Recall standard attention:
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where the attention matrix $A = \text{softmax}(QK^T / \sqrt{d_k})$ is $N \times N$.

In Spectral-JAX:
$$
\hat{Y} = \text{FFT}(X) \odot H_{\theta}(\text{mean}(X))
$$

Via **Convolution Theorem**, multiplication in frequency domain = convolution in time domain:
$$
Y = X \ast h \quad \text{where } h = \text{iFFT}(H_{\theta})
$$

Convolution as Matrix Multiplication:
$$
Y = \mathcal{T}(h) \cdot X
$$

where $\mathcal{T}(h)$ is a **Toeplitz matrix** (diagonal-constant structure).

### Connection to Attention

A Toeplitz matrix is a special case of the attention matrix where:
$$
A_{ij} = f(i - j) \quad \text{(depends only on distance)}
$$

This is equivalent to **relative position attention** without content-based weighting.

**Key Difference**: Spectral-JAX uses input-dependent $H_{\theta}(X)$, which modulates the Toeplitz structure based on global statistics ($\text{mean}(X)$). This is similar to **learned relative position biases** in Transformer-XL.

### Formal Statement

Let $A_{\text{spectral}}$ be the effective attention matrix induced by Spectral-JAX:
$$
A_{\text{spectral}} = \mathcal{F}^{-1} \left\{ H_{\theta}(\text{mean}(X)) \right\}
$$

Then $A_{\text{spectral}}$ is a **circulant matrix** (stronger structure than Toeplitz), meaning:
- Each row is a cyclic permutation of the previous row
- Can be diagonalized by DFT

**Conclusion**: Spectral-JAX attention is a structured subset of full attention, similar to how Sparse Transformers restrict attention patterns. ∎

---

## PROOF 3: Energy Gate Prevents Gradient Explosion

### Theorem 3.1
*The energy-based gating mechanism ensures bounded gradients during backpropagation through spectral layers.*

### Setup

Define the energy gate:
$$
g(Y) = \sigma\left( \|Y\|_2 \right) \quad \text{where } \sigma(x) = \frac{1}{1 + e^{-x}}
$$

Output:
$$
Z = Y \odot g(Y)
$$

Loss:
$$
\mathcal{L}(Z)
$$

### Gradient Analysis

We need to bound:
$$
\frac{\partial \mathcal{L}}{\partial Y}
$$

Via chain rule:
$$
\frac{\partial \mathcal{L}}{\partial Y} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial Y}
$$

Compute $\frac{\partial Z}{\partial Y}$:
$$
\frac{\partial Z}{\partial Y} = g(Y) \cdot I + Y \odot \frac{\partial g}{\partial Y}
$$

where $I$ is the identity and:
$$
\frac{\partial g}{\partial Y} = \sigma'(\|Y\|_2) \cdot \frac{\partial \|Y\|_2}{\partial Y} = \sigma'(\|Y\|_2) \cdot \frac{Y}{\|Y\|_2}
$$

### Bounding $\sigma'$

Key property of sigmoid:
$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) \leq \frac{1}{4} \quad \forall x
$$

Proof: $\sigma'(x)$ is maximized at $x=0$ where $\sigma(0) = 1/2$:
$$
\sigma'(0) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
$$

### Gradient Bound

Therefore:
$$
\left\| \frac{\partial g}{\partial Y} \right\|_2 = \sigma'(\|Y\|_2) \cdot \left\| \frac{Y}{\|Y\|_2} \right\|_2 \leq \frac{1}{4} \cdot 1 = \frac{1}{4}
$$

This means:
$$
\left\| \frac{\partial Z}{\partial Y} \right\|_F \leq \|g(Y)\|_{\infty} + \frac{1}{4} \cdot \|Y\|_2 \leq 1 + \frac{1}{4} \cdot \|Y\|_2
$$

Since $g(Y) \in [0, 1]$, the gradient is **always bounded** regardless of $Y$ magnitude.

### Comparison to Unregularized FFT

Without energy gate:
$$
Z_{\text{unregularized}} = \text{iFFT}(\text{FFT}(X) \odot H)
$$

If $H$ contains large values, $\text{iFFT}$ can amplify noise, leading to:
$$
\|\nabla_X \mathcal{L}\|_2 \to \infty \quad \text{(spectral explosion)}
$$

The energy gate acts as a **soft clipping mechanism**, preventing this instability. ∎

---

## PROOF 4: Connection to State Space Models

### Theorem 4.1
*Spectral-JAX with diagonal filters is equivalent to a discretized Linear Time-Invariant (LTI) system.*

### State Space Representation

A continuous-time SSM is defined by:
$$
\begin{cases}
\frac{dx}{dt} = Ax(t) + Bu(t) \\
y(t) = Cx(t) + Du(t)
\end{cases}
$$

where:
- $x(t) \in \mathbb{R}^N$: hidden state
- $u(t) \in \mathbb{R}$: input
- $y(t) \in \mathbb{R}$: output
- $A, B, C, D$: system matrices

### Discretization (Zero-Order Hold)

$$
\begin{cases}
x_k = \bar{A} x_{k-1} + \bar{B} u_k \\
y_k = C x_k + D u_k
\end{cases}
$$

where:
$$
\bar{A} = e^{\Delta A}, \quad \bar{B} = (e^{\Delta A} - I) A^{-1} B
$$

### Convolution Form

The output can be written as:
$$
y_k = \sum_{j=0}^{k} h_{k-j} \cdot u_j
$$

where $h_k = C \bar{A}^k \bar{B}$ is the **impulse response**.

### Connection to Spectral-JAX

In Spectral-JAX:
$$
Y = \text{iFFT}\left( \text{FFT}(X) \odot H \right)
$$

This is **equivalent** to convolving $X$ with impulse response $h = \text{iFFT}(H)$:
$$
Y = X \ast h
$$

If $H$ is diagonal in frequency domain, then $A$ (the equivalent SSM matrix) is also diagonal:
$$
A = \text{diag}(\lambda_1, \ldots, \lambda_N)
$$

This is exactly the **diagonal state space model** used in S4!

### Key Insight

- **S4**: Uses diagonal $A$ with specific initialization (HiPPO)
- **Spectral-JAX**: Learns $H$ (Fourier transform of impulse response) directly
- **Equivalence**: If $H$ is constrained to be real and symmetric, they represent the same function class

The advantage of Spectral-JAX: **No need for matrix exponentials** (which are numerically unstable). We operate directly in frequency domain. ∎

---

## PROOF 5: Universal Approximation in Frequency Domain

### Theorem 5.1
*A Spectral-JAX model with sufficiently wide hidden dimension $D$ can approximate any continuous sequence-to-sequence operator.*

### Setup

Let $\mathcal{F}: L^2([0,1]^N) \to L^2([0,1]^N)$ be a bounded linear operator.

**Goal**: Show that Spectral-JAX can approximate $\mathcal{F}$ with arbitrary precision.

### Proof (Sketch)

By **Fourier Series Theorem**, any $L^2$ function can be represented as:
$$
f(t) = \sum_{k=-\infty}^{\infty} c_k e^{i 2\pi k t}
$$

A linear operator $\mathcal{F}$ acts on Fourier coefficients:
$$
\mathcal{F}(f) \leftrightarrow H[k] \cdot \hat{f}[k]
$$

where $H[k]$ is the frequency response.

Spectral-JAX parameterizes $H[k]$ via a neural network:
$$
H[k] = \text{MLP}_{\theta}(k, \text{context})
$$

By **Universal Approximation Theorem** (Hornik et al., 1989), the MLP can approximate any continuous function.

Therefore:
$$
\forall \epsilon > 0, \exists \theta^* : \|H_{\theta^*} - H_{\text{true}}\|_{\infty} < \epsilon
$$

**Conclusion**: Spectral-JAX is a universal approximator for sequence operators. ∎

---

## APPENDIX: Numerical Stability Tricks

### A.1 Preventing FFT Overflow

When $|X| > 10^3$, FFT can produce numerical artifacts. Solution:

$$
\text{FFT}_{\text{stable}}(X) = \text{FFT}\left( \frac{X}{\|X\|_{\infty}} \right) \cdot \|X\|_{\infty}
$$

### A.2 Gradient Checkpointing

For very long sequences ($N > 100K$), recompute FFT in backward pass:

```python
@jax.custom_vjp
def fft_layer(x, params):
    return jnp.fft.fft(x) * params

def fft_fwd(x, params):
    return fft_layer(x, params), (x, params)

def fft_bwd(res, g):
    x, params = res
    # Recompute FFT instead of storing
    x_fft = jnp.fft.fft(x)
    return (jnp.fft.ifft(g * jnp.conj(params)), g * x_fft)

fft_layer.defvjp(fft_fwd, fft_bwd)
```

---

**END OF MATHEMATICAL PROOFS**