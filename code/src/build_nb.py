"""Build the Kaggle notebook from the percent-format source; the '[writefile X]' cell gets the module file."""
import sys
from pathlib import Path
import nbformat

src, module, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
gpu = len(sys.argv) < 5 or sys.argv[4] != "cpu"
cells, kind, buf, wf = [], None, [], None


def flush():
    if kind is None:
        return
    body = list(buf)
    while body and not body[-1].strip():
        body.pop()
    while body and not body[0].strip():
        body.pop(0)
    if kind == "md":
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body)))
    elif kind == "wf":
        cells.append(nbformat.v4.new_code_cell(f"%%writefile {wf}\n" + module.read_text(encoding="utf-8").rstrip("\n")))
    else:
        cells.append(nbformat.v4.new_code_cell("\n".join(body)))


for line in src.read_text(encoding="utf-8").splitlines():
    if line.startswith("# %%"):
        flush()
        buf = []
        if "[markdown]" in line:
            kind = "md"
        elif "[writefile" in line:
            kind, wf = "wf", line.split("[writefile", 1)[1].strip(" ]")
        else:
            kind = "code"
    else:
        buf.append(line)
flush()
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "kaggle": ({"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                           "language": "python", "sourceType": "notebook"} if gpu else
                          {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False,
                           "language": "python", "sourceType": "notebook"})}
nbformat.validate(nb)
nbformat.write(nb, out)
print(f"wrote {out}: {len(cells)} cells ({sum(c.cell_type == 'code' for c in cells)} code)")
