
# muxsim

a python module for generate cell / guide matrices for demultiplex testing.

## Installation

```bash
pip install muxsim
```

## Usage

Muxsim is expected to be used as a python module. It has reasonable defaults for most use cases, but can be configured to your liking.

Here is a simple example of how to use muxsim:

```python
from muxsim import MuxSim

ms = MuxSim()
matrix = ms.sample()
```

The sampling scheme can be fully parameterized like so:

```python
from muxsim import MuxSim

ms = MuxSim(
    num_cells=10000,
    num_guides=100,
    n=10.0,
    p=0.1,
    λ=0.8,
    random_state=42,
)
matrix = ms.sample()
```

## Methods

The simulator, `muxsim`, is based on a Multinomial distribution, where the number of draws $(\mathcal{S})$ of cell $(i)$ is drawn from a Negative Binomial distribution representing the number of observed UMIs of that cell.

$$
\begin{align}
\mathbb{U}_i &\sim \text{Multinomial}(\mathbb{S}_i,\ f_i) \\
\mathbb{S}_i &\sim \text{NegativeBinomial}(n,\ p) \\
\end{align}
$$

The frequencies of the multinomial distribution are cell specific and sum to 1:

$$
\sum_{j=1}^{M}{f_{ij}} = 1
$$

The background frequencies are assumed to be equiprobable (where $`(\forall u,v \in M)(f_{iu} = f_{iv})`$ ), except for signal guides - which would be a scaled by some value $(r)$. The number of signal guides is chosen using a Poisson prior to simulate situations where the expected multiplicity of infection (MOI) can change:

$$
\mathbb{I}_i \sim \text{Poisson}(\lambda)
$$


The signal guides are chosen randomly from the guide set where the number of choices is equal to the MOI of that cell:

$$
\mathbb{C}_i \sim \text{Uniform}(M, \mathbb{I}_i)
$$

This allows us to then set the the signal guides at a rate $(r)$ above the background with the following expression:

$$
t_{ij} = 
\begin{cases}
r,& \text{if } j \in \mathbb{C}_i \\
1,& \text{if } j \not\in \mathbb{C}_i
\end{cases}
$$

Which can then be turned into the frequency matrix:

$$
f_i = \frac{t_{i}}{\sum_{j=1}^{M}t_{ij}}
$$

Which forces $f_{ij}$ to sum to 1. 