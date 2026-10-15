"""Helper to build and execute Jupyter notebooks from (kind, source) cell lists."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
import os

def build(path, cells, title):
    nb = new_notebook()
    nb.cells = [
        (new_markdown_cell(src) if kind == "md" else new_code_cell(src))
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    with open(path, "w") as f:
        nbf.write(nb, f)
    return nb

def execute(path, timeout=1200):
    nb = nbf.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path) or "."}})
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("executed:", path)
