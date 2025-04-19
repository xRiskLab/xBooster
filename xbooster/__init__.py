"""
xbooster
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xbooster")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
