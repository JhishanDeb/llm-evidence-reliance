"""run_memcheck with the mock backend: normal run, empty responses (tabmemcheck raises inside header_test), a
tiny context (every prompt too long), and a model that copies the true next row (must be flagged)."""
import sys

sys.path.insert(0, sys.argv[1])
import ler_main as L  # noqa: E402

data = sys.argv[2]
files = {"physionet2012": f"{data}/physionet2012/Outcomes-a.txt",
         "physionet2019": f"{data}/physionet2019/training_setA/p000001.psv",
         "diabetes130": f"{data}/diabetes130/diabetic_data.csv"}
quiet = lambda *a: None


class Echo(L.MockBackend):
    """Returns '' (empty) or, in 'copy' mode, the next row of the file after the last prompt line it recognises."""

    def __init__(self, mode, rows):
        super().__init__([])
        self.mode, self.rows = mode, rows

    def generate_text(self, token_id_lists, max_tokens):
        if self.mode == "empty":
            return [""]
        p = self.last_prompt
        for i, r in enumerate(self.rows[:-1]):
            if p.rstrip().endswith(r):
                return ["\n" + self.rows[i + 1] + "\n"]
        return ["0"]


class Tok(L._MockTok):
    def encode(self, text, add_special_tokens=False):
        Tok.last = text
        return list(range(len(text) // 4 + 1))


for kind in ("instruct", "base"):
    be = L.MockBackend([])
    res = L.run_memcheck(files, L.TabmemLLM(be, kind, 16384), log=quiet)
    for ds, r in res.items():
        assert r["header"]["status"] == "ok" and r["row"]["status"] == "ok", (kind, ds, r["header"], r["row"])
        assert len(r["header"]["split_rows"]) == 4
    print(kind, "normal run ok:", {ds: (r["header"]["matched_chars"], r["row"]["exact_matches"]) for ds, r in res.items()})

be = Echo("empty", [])
res = L.run_memcheck(files, L.TabmemLLM(be, "instruct", 16384), log=quiet)
print("empty responses:", {ds: (r["header"]["status"][:60], r["row"]["status"][:40]) for ds, r in res.items()})

res = L.run_memcheck(files, L.TabmemLLM(L.MockBackend([]), "instruct", 300), log=quiet)
assert all(r["header"]["status"].startswith("not run") for r in res.values())
assert all(r["row"]["status"].startswith("prompt too long") for r in res.values())
print("tiny context: header and row tests reported as not run / too long, no crash")

# a 'model' that completes rows verbatim in completion mode must be flagged
rows = L._csv_rows(files["physionet2012"])
be = Echo("copy", rows)
be.tokenizer = Tok()
orig = be.generate_text
be.generate_text = lambda ids, mt: (setattr(be, "last_prompt", Tok.last), orig(ids, mt))[1]
res = L.run_memcheck({"physionet2012": files["physionet2012"]}, L.TabmemLLM(be, "base", 16384), log=quiet)
r = res["physionet2012"]
print("copying model:", r["row"]["exact_matches"], "/", r["row"]["num_queries"], "flagged", r["flag_verbatim_rows"])
assert r["flag_verbatim_rows"] and r["row"]["exact_matches"] == r["row"]["num_queries"]
print("memcheck tests passed")
