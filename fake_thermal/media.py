"""Locate media tools in GUI-launched, per-user installations too."""
from pathlib import Path
import shutil
import sys


def media_tool(name):
    found = shutil.which(name)
    if found:
        return found
    adjacent = Path(sys.executable).parent / (name + ('.exe' if sys.platform == 'win32' else ''))
    return str(adjacent) if adjacent.is_file() else name
