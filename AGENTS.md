# AGENTS.md — sdypy-model contributor & agent guide

Guidance for anyone (human or AI) developing **sdypy-model**. Claude Code reads
it via a one-line `CLAUDE.md` (`@AGENTS.md`). The org-wide rules (development
workflow, SEP governance, naming, canonical specs) live in the sdypy hub and are
linked at the end, never restated here.

## What this package is

System modeling: the `model` portion of the `sdypy` namespace (PEP 420,
`from sdypy import model`). It covers lumped-mass models and finite elements
(beam, shell, tetrahedron meshes, acoustic external problems). The public API
is the curated `__all__` in `sdypy/model/__init__.py`.

## Development environment

Use `uv`. Supported Python versions: `requires-python` in `pyproject.toml`.

```console
uv venv && uv pip install -e ".[dev,qt]"   # dev install; `qt` adds the Qt binding
QT_QPA_PLATFORM=offscreen pytest           # headless tests need the offscreen Qt platform
python -m build                            # sdist + wheel
```

## Org-wide rules (sdypy hub)

Copied verbatim from the hub's `AGENTS.md`; edit it there. The hub's template
checker compares this copy.

<!-- >>> sdypy hub links -->
- [AGENTS.md](https://github.com/sdypy/sdypy/blob/main/AGENTS.md) — workflow and definition of done
- [AGENTS.md § Releasing](https://github.com/sdypy/sdypy/blob/main/AGENTS.md#releasing) — the only way to release
- [docs/seps/](https://github.com/sdypy/sdypy/tree/main/docs/seps/) — SEP governance docs
- [docs/source/dev/nomenclature.rst](https://github.com/sdypy/sdypy/blob/main/docs/source/dev/nomenclature.rst) — naming a new public term (SEP 2)
- [openspec/specs/](https://github.com/sdypy/sdypy/tree/main/openspec/specs/) — canonical capability specs
- [REQUIREMENTS.md](https://github.com/sdypy/sdypy/blob/main/REQUIREMENTS.md) — requirements roster
<!-- <<< sdypy hub links -->
