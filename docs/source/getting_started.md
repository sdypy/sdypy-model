# Getting started

<!-- TODO: short intro — what sdypy-model is and what this page covers. -->

## Installation

`sdypy-model` is part of the [SDyPy](https://github.com/sdypy) namespace and is
published on [PyPI](https://pypi.org/project/sdypy-model/). It requires
**Python ≥ 3.10**.

Install the latest release with `pip`:

```console
$ pip install sdypy-model
```

Alternatively, install the full [`sdypy`](https://pypi.org/project/sdypy/)
umbrella package, which pulls in `sdypy-model` together with the other SDyPy
subpackages (`sdypy-EMA`, `sdypy-FRF`, `sdypy-io`, ...):

```console
$ pip install sdypy
```

The package is imported through the shared `sdypy` namespace:

```python
import sdypy as sd

sd.model.Beam        # FEM elements
sd.model.Shell
sd.model.Tetrahedron
sd.model.AcousticExternalProblem   # exterior acoustic BEM
```

### Optional dependencies

Some extras are only needed for running the bundled examples or for development:

```console
$ pip install "sdypy-model[examples]"   # adds matplotlib for the example scripts
$ pip install "sdypy-model[dev]"        # docs + test tooling (sphinx, pytest, ...)
```

### Installing from source

To work with the development version (or contribute), clone the repository and
install it in editable mode:

```console
$ git clone https://github.com/sdypy/sdypy-model.git
$ cd sdypy-model
$ pip install -e .
```

## A minimal example

The snippet below builds a simple beam finite-element model, solves its
eigenproblem for the natural frequencies and mode shapes, and plots the first
mode. All quantities use a consistent **mm / N** unit system, so the natural
frequencies come out in Hz.

```python
import numpy as np
import matplotlib.pyplot as plt
from sdypy.model import Beam

# A 20-element beam (consistent mm / N unit system)
n_elements = 20
length  = np.full(n_elements, 500 / n_elements)   # element lengths [mm]
density = np.full(n_elements, 7850e-12)           # density [t/mm^3]
Young   = np.full(n_elements, 180e3)              # Young's modulus [N/mm^2]

# org/conec (node coordinates and connectivity) are auto-generated from n_nodes
beam = Beam(org=None, conec=None,
            length=length, width=30, height=15,   # rectangular cross-section [mm]
            density=density, Young=Young,
            n_nodes=n_elements + 1)

nat_freq, modes = beam.solve()                    # natural frequencies [Hz], mode shapes
print(nat_freq[:6])

# The beam is unconstrained, so the first two modes are rigid-body modes (~0 Hz).
# Plot the first elastic (bending) mode instead.
mode = 2
plt.plot(modes[::2, mode])                        # translational DOFs are every other entry
plt.title(f"First elastic mode — {nat_freq[mode]:.1f} Hz")
plt.xlabel("node")
plt.ylabel("displacement")
plt.show()
```

```{note}
Plotting requires `matplotlib`, which is installed with the `[examples]` extra
(`pip install "sdypy-model[examples]"`).
```

## Where to go next

- {doc}`Tutorials <tutorials/index>` — worked examples for each element type:
  beam, shell, tetrahedron, and the acoustic BEM.
- {doc}`Code documentation <code>` — the full API reference.
- {doc}`Examples <examples>` — runnable scripts shipped with the package.
- [GitHub repository](https://github.com/sdypy/sdypy-model) — source code,
  issues, and contributing.
