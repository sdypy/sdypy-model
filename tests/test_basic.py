import subprocess
import sys


def test_namespace_import():
    """`import sdypy as sd` must expose `sd.model` and the public entry points.

    This is the documented top-level API used throughout the README and docs
    (`import sdypy as sd; sd.model.AcousticExternalProblem`). It only works
    because ``sdypy/__init__.py`` does ``from . import model``; guard against
    that file being removed again. Run in a fresh interpreter so the result
    does not depend on another test having already imported ``sdypy.model``.
    """
    code = (
        "import sdypy as sd; "
        "assert hasattr(sd, 'model'); "
        "assert sd.model.AcousticExternalProblem is not None; "
        "assert sd.model.Beam is not None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
