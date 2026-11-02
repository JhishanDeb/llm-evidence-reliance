"""Execute a notebook offline (CPU tests): python run_nb.py NOTEBOOK WORKDIR OUT_NOTEBOOK. Prints OK or the error."""
import os
import sys

import nbformat
from nbclient import NotebookClient

nb_path, workdir, out_path = sys.argv[1:4]
os.makedirs(workdir, exist_ok=True)
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": workdir}})
status = "OK"
try:
    client.execute()
except Exception as e:
    status = f"FAILED: {type(e).__name__}: {str(e)[-3000:]}"
finally:
    nbformat.write(nb, out_path)
print(status)
