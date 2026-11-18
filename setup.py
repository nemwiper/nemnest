"""Only the optional compiled part lives here; everything else is in pyproject.toml.

`_runs.c` is a plain C function that nemnest loads with ctypes. If no C compiler
is available the install still succeeds and the Python decoder is used."""
from setuptools import Extension, setup

setup(ext_modules=[Extension("nemnest._runs", sources=["src/nemnest/_runs.c"], optional=True)])
