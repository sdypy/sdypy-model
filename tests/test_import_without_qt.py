import subprocess
import sys


def test_import_without_qt():
    """`sdypy.model` imports when no Qt binding is installed (no `qt` extra)."""
    code = (
        "import sys\n"
        "for name in ('pyvistaqt', 'PyQt6', 'PyQt5', 'PySide6', 'PySide2', 'qtpy'):\n"
        "    sys.modules[name] = None\n"
        "import sdypy.model\n"
        "from sdypy.model import mesh\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
