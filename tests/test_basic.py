import subprocess
import sys


def test_namespace_import():
    """`sdypy.model` imports as a namespace portion and exposes its entry points.

    ``sdypy-model`` ships no ``sdypy/__init__.py`` (SEP 3); ``import sdypy as
    sd; sd.model`` is provided by the ``sdypy`` umbrella and tested there. Run
    in a fresh interpreter so the result does not depend on another test
    having already imported ``sdypy.model``.
    """
    code = (
        "import sdypy.model as m; "
        "assert m.AcousticExternalProblem is not None; "
        "assert m.Beam is not None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
