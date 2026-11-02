"""Checks that the registered analysis recovers the effects built into the fake models of run_mock_v4.sh.
usage: python check_mock_v4.py RESULTS_JSON"""
import json
import sys

R = json.load(open(sys.argv[1]))
C, F, K = R["cells"], R["families"], R["confirmatory"]
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def W(cell, pair, part="W"):
    c = C[cell]
    comp = c["registered"] if pair == "registered" else c["pairs"][pair]
    return comp["pairs"][pair][part]


def neg(r):
    return r["conclusive"] and r["sign"] == -1


def pos(r):
    return r["conclusive"] and r["sign"] == 1


def small(r, lim=0.3):
    return abs(r["a"]["est"]) < lim and abs(r["a_prime"]["est"]) < lim


DS = ("physionet2012", "physionet2019", "diabetes130")
for ds in DS:
    k = f"llama-instruct|{ds}"
    check(f"{k}: W < 0 with every pair, O < 0 (both versions), W near 0 in plain text",
          all(neg(W(k, p)) for p in ("registered", "tf", "neg", "order", "para")) and W(k, "registered", "O")["sign"] == -1
          and small(W(k, "plain")))
    k = f"qwen-instruct|{ds}"
    check(f"{k}: W < 0 with Yes/No, near 0 with True/False and reduced; plain format invalid",
          neg(W(k, "registered")) and small(W(k, "tf")) and pos(C[k]["pairs"]["tf"]["contrasts"]["dW"])
          and neg(W(k, "neg")) and not C[k]["validity"]["pairs"]["plain"]["passes"])
    k = f"olmo3-instruct|{ds}"
    check(f"{k}: named outcome: W < 0 with True/False, near 0 with the negated question", neg(W(k, "registered"))
          and neg(W(k, "tf")) and small(W(k, "neg")) and pos(C[k]["pairs"]["neg"]["contrasts"]["dW"]))
    k = f"falcon3-7b|{ds}"
    check(f"{k}: W > 0; O < 0 with set A and O > 0 with the alarming-last order", pos(W(k, "registered"))
          and pos(W(k, "order")) and neg(W(k, "registered", "O")) and pos(W(k, "order", "O")))
    k = f"yi15-9b|{ds}"
    check(f"{k}: W > 0, near 0 in the paraphrased prompt", pos(W(k, "registered")) and small(W(k, "para")))
check("R8 and R9: W kept in both; joint reading 'answer polarity'", K["R8"]["W"]["result"] == "kept"
      and K["R9"]["W"]["result"] == "kept" and K["joint_reading"] == "answer polarity")
check("R8: Qwen2.5 on PhysioNet 2019 reduced with True/False; R9: both Olmo cells reduced",
      K["R8"]["W"]["cells"]["qwen-instruct|physionet2019"]["reduced"]
      and all(K["R9"]["W"]["cells"][f"olmo3-instruct|{d}"]["reduced"] for d in DS[:2]))
check("R12: both Yi cells not supported and reduced in the paraphrased prompt",
      all(not K["R12"]["W"]["cells"][f"yi15-9b|{d}"]["supported"] and K["R12"]["W"]["cells"][f"yi15-9b|{d}"]["reduced"]
          for d in DS[:2]))
for d in ("physionet2012", "diabetes130"):
    f = F[f"llama|{d}"]["primary"]
    check(f"llama family {d}: W near 0 in B.plain, < 0 in B.chat and I.chat; P near 0, F < 0",
          small(f["pairs"]["B_plain"]["W"]) and neg(f["pairs"]["B_chat"]["W"]) and neg(f["pairs"]["I_chat"]["W"])
          and small(f["contrasts"]["P"]) and neg(f["contrasts"]["F"]))
f = F["qwen|physionet2019"]["primary"]
check("qwen family PhysioNet 2019: W near 0 in both base formats, P < 0, F near 0",
      small(f["pairs"]["B_plain"]["W"]) and small(f["pairs"]["B_chat"]["W"]) and neg(f["contrasts"]["P"])
      and small(f["contrasts"]["F"]))
r10 = K["R10"]
check("R10: P supported in 1 of 4 cells (Qwen2.5, PhysioNet 2019), F in the direction of W in 2 (Llama 3.1) -> "
      "'not separated by this study'", r10["n_eligible"] == 4 and r10["n_P_supported"] == 1
      and r10["P"]["qwen|physionet2019"]["supported"] and r10["n_F_in_direction"] == 2 and not r10["supported"]
      and r10["interpretation"] == "not separated by this study")
check("R10: secondary comparison with I.plain only for Llama 3.1 (Qwen2.5's plain format failed validity)",
      F["llama|physionet2012"]["secondary"] is not None and F["llama|physionet2012"]["validity"]["I_plain"]["passes"]
      and not F["qwen|physionet2012"]["validity"]["I_plain"]["passes"])
check("R13: the cue moves the fake models more than the net-number card in every valid cell",
      K["R13"]["supported"] and K["R13"]["n_supported"] == K["R13"]["n_valid"] == 15)
g = R["generated"]
check("generated answers: prefill fully parsed and in agreement except for Yi-1.5 (built-in flips)",
      all(g[t]["prefill"]["parsed_share"] == 1 for t in g) and all(g[t]["prefill"]["agreement"] == 1 for t in g if t != "yi15-9b")
      and g["yi15-9b"]["prefill"]["agreement"] < 1 and all(0.6 < g[t]["free"]["parsed_share"] < 0.95 for t in g))
check("static lean computed for every tested pair", all(R["lean"][p]["n_valid"] >= 12 for p in ("registered", "tf", "neg", "order", "para")))
print(f"\n{sum(ok)} of {len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
