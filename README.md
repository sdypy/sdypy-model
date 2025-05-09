# `sdypy-model`

A namespace Python package for the SDyPy project.

The package currently contains the following finite elements:

- `beam`: Euler-Bernoulli and Timoshenko beam elements.
- `shell`: MITC4 shell elements.
- `tet`: Quadratic (10 node) tetrahedral elements.

**Note**: The package is still under development and may not be fully functional.

## Beam elements

```python
import sdypy as sd

beam_obj = sd.model.Beam(...)

M = beam_obj.M
K = beam_obj.K
```

## Shell elements

```python
import sdypy as sd

shell_obj = sd.model.Shell(...)

M = shell_obj.M
K = shell_obj.K
```

## Tetrahedral elements
```python

import sdypy as sd

tet_obj = sd.model.Tetrahedron(...)

M = tet_obj.M
K = tet_obj.K
```

# Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Use of this software is at your own risk.

This software is distributed under the terms of the MIT License. See the LICENSE file for more details.