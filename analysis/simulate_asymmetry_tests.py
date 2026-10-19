"""Rejection rates of the asymmetry tests (Deviation D1 in PREREGISTRATION.md).

simulate_asymmetry_measures.py shows the bias of each asymmetry measure. This
script shows what matters for H3: how often each bootstrap test declares an
asymmetry toward alarm (estimate > 0 and two-sided bootstrap p < 0.05) when
there is none (false positives; the nominal rate is 0.025) and when there is
one (power).

Simulated world (600 patients per study):
  y      true outcome, half positive
  z      XGBoost log-odds, 1.2 * (2y - 1) + N(0, 1.3)
  mu     the model's baseline log-odds: lean + 3z + r, where the residual r
         (the model's own reading of the patient) is normal, right-skewed,
         left-skewed or bimodal (80/20), all with SD 4; in the 'tanh' setting
         the mean is lean + 6 tanh(z) instead of lean + 3z (a first stage that
         is not cubic) with a normal residual
  t      true card direction, alarm with probability 1 / (1 + exp(-1.5 (z - 1)))
  l0     the no-evidence reading: mu plus an idiosyncratic error with SD eta
         (0, 1 or 2) that only the no-evidence prompt has
  la, ls the readings under the alarm-pointing and the safe-pointing card, from
         one of four response rules (add, adopt, switch, saturate; see
         simulate_asymmetry_measures.py) plus N(0, 1); true cards are followed
         50% more than flipped cards; 'alarm' rows make alarm-pointing cards 1.5
         times as strong

Tests (paired bootstrap over patients, 400 resamples here; the study uses
10,000):
  delta    A - S (registered primary measure)
  logodds  unadjusted log-odds asymmetry (registered secondary measure)
  a        the asymmetry on the expected baseline from z, z^2, z^3, y
  a_prime  the same second stage on the no-evidence reading l0 itself
  iut      the H3 test of Deviation D1: both a and a_prime above 0, with
           p = max(p_a, p_a_prime) (intersection-union test)

The bootstrap is vectorised (weighted least squares with resample counts as
weights). Before the main loop the script checks, on one simulated study, that
this gives the same estimates as refitting on resampled rows.

Run:  python analysis/simulate_asymmetry_tests.py
Runtime: about one hour on two CPU cores. The output is deterministic (seed 2026).
"""
import sys
import time

import numpy as np

N_PATIENTS = 600
N_REPS = 150
N_BOOT = 400
SEED = 2026
TESTS = ("delta", "logodds", "a", "a_prime", "iut")


def _wls_coef(X, d, W):
    """X: n x k, d: n, W: B x n resample counts -> B x k weighted least squares coefficients."""
    XtWX = np.einsum("bn,nk,nl->bkl", W, X, X)
    XtWd = np.einsum("bn,nk,n->bk", W, X, d)
    return np.linalg.solve(XtWX, XtWd[..., None])[..., 0]


def residual(rng, shape, n):
    if shape in ("normal", "tanh"):
        return rng.normal(0, 4, n)
    if shape in ("right", "left"):
        g = (rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)
        return 4 * (g if shape == "right" else -g)
    if shape == "bimodal":
        hi = rng.random(n) < 0.2
        x = np.where(hi, rng.normal(6.0, 2.0, n), rng.normal(-1.5, 2.0, n))
        return 4 * (x - x.mean()) / x.std()
    raise ValueError(shape)


def simulate(rng, response, shape, lean, alarm_factor, eta, truth=0.5, n=N_PATIENTS):
    y = (np.arange(n) % 2).astype(float)
    z = 1.2 * (2 * y - 1) + rng.normal(0, 1.3, n)
    mu = lean + (6 * np.tanh(z) if shape == "tanh" else 3 * z) + residual(rng, shape, n)
    t = np.where(rng.random(n) < 1 / (1 + np.exp(-1.5 * (z - 1))), 1.0, -1.0)
    l0 = mu + (rng.normal(0, eta, n) if eta > 0 else 0.0)

    def respond(card):
        k = np.where(card > 0, alarm_factor, 1.0) * (1 + truth * (card == t))
        if response == "add":
            out = mu + card * 3 * k
        elif response == "adopt":
            w = np.clip(0.3 * k, 0, 1)
            out = (1 - w) * mu + w * card * 8
        elif response == "switch":
            q = np.clip(0.4 * k, 0, 1)
            out = np.where(rng.random(n) < q, card * 8, mu)
        elif response == "saturate":
            out = mu + card * 3 * k * (1 - np.tanh(card * mu / 5))
        else:
            raise ValueError(response)
        return out + rng.normal(0, 1, n)

    return y, z, t, l0, respond(np.ones(n)), respond(-np.ones(n))


def estimates(y, z, t, l0, la, ls, W):
    """Row b of W holds the resample counts; returns a dict of arrays (one estimate per row)."""
    n = len(y)
    d = (la - l0) + (ls - l0)
    zs = (z - z.mean()) / z.std()
    Z = np.column_stack([np.ones(n), zs, zs ** 2, zs ** 3, y])
    E = _wls_coef(Z, l0, W) @ Z.T                              # first stage, one fit per resample
    s = E[0].std()
    a = np.array([_wls_coef(np.column_stack([np.ones(n), E[b], (E[b] / s) ** 3, t]), d, W[b:b + 1])[0, 0]
                  for b in range(len(W))])
    s0 = l0.std()
    a_prime = _wls_coef(np.column_stack([np.ones(n), l0, (l0 / s0) ** 3, t]), d, W)[:, 0]
    c0, pa, ps = l0 > 0, la > 0, ls > 0
    delta = (W * (pa & ~c0)).sum(1) / (W * ~c0).sum(1) - (W * (~ps & c0)).sum(1) / (W * c0).sum(1)
    logodds = (W * (la - l0)).sum(1) / W.sum(1) - (W * (l0 - ls)).sum(1) / W.sum(1)
    return {"delta": delta, "logodds": logodds, "a": a, "a_prime": a_prime}


def boot_test(v):
    """v[0] = full-sample estimate, v[1:] = bootstrap; -> (rejects toward alarm, p)."""
    b = v[1:]
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return bool(p < 0.05 and v[0] > 0), p


def run_tests(sim, rng):
    n = len(sim[0])
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)]).astype(float)
    est = estimates(*sim, W)
    out = {k: boot_test(v) for k, v in est.items()}
    p_iut = max(out["a"][1], out["a_prime"][1])
    out["iut"] = (bool(p_iut < 0.05 and est["a"][0] > 0 and est["a_prime"][0] > 0), p_iut)
    return out


def self_check():
    """The vectorised bootstrap equals refitting both stages on resampled rows."""
    rng = np.random.default_rng(1)
    y, z, t, l0, la, ls = simulate(rng, "saturate", "right", 3.0, 1.5, 1.0)
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(3)]
    W = np.vstack([np.bincount(i, minlength=n) for i in idx]).astype(float)
    est = estimates(y, z, t, l0, la, ls, W)
    for k, i in enumerate(idx):
        yy, zz, tt, l00, laa, lss = y[i], z[i], t[i], l0[i], la[i], ls[i]
        d = (laa - l00) + (lss - l00)
        zs = (zz - zz.mean()) / zz.std()
        Z = np.column_stack([np.ones(n), zs, zs ** 2, zs ** 3, yy])
        e = Z @ np.linalg.lstsq(Z, l00, rcond=None)[0]
        a = np.linalg.lstsq(np.column_stack([np.ones(n), e, (e / e.std()) ** 3, tt]), d, rcond=None)[0][0]
        a0 = np.linalg.lstsq(np.column_stack([np.ones(n), l00, (l00 / l00.std()) ** 3, tt]), d, rcond=None)[0][0]
        assert abs(a - est["a"][k]) < 1e-8 and abs(a0 - est["a_prime"][k]) < 1e-8, (a, est["a"][k], a0, est["a_prime"][k])
    print("Self-check passed: the vectorised bootstrap matches refitting on resampled rows.\n")


def main():
    self_check()
    rng = np.random.default_rng(SEED)
    print(f"Share of {N_REPS} simulated studies (600 patients each, {N_BOOT} bootstrap resamples) in which each test "
          "declares an asymmetry toward alarm.\n'none' rows: false positives (nominal 0.025). 'alarm' rows: power.\n")
    print("| response | asymmetry | residual | eta | lean | " + " | ".join(TESTS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(TESTS))
    worst = {k: 0.0 for k in TESTS}
    weakest = {k: 1.0 for k in TESTS}
    t0 = time.time()
    for response in ("add", "adopt", "switch", "saturate"):
        for asym, factor in (("none", 1.0), ("alarm", 1.5)):
            for shape in ("normal", "right", "left", "bimodal", "tanh"):
                for eta in (0.0, 1.0, 2.0):
                    for lean in (-3.0, 3.0):
                        rej = {k: 0 for k in TESTS}
                        for _ in range(N_REPS):
                            r = run_tests(simulate(rng, response, shape, lean, factor, eta), rng)
                            for k in TESTS:
                                rej[k] += r[k][0]
                        rate = {k: rej[k] / N_REPS for k in TESTS}
                        for k in TESTS:
                            if asym == "none":
                                worst[k] = max(worst[k], rate[k])
                            else:
                                weakest[k] = min(weakest[k], rate[k])
                        print(f"| {response} | {asym} | {shape} | {eta:.0f} | {lean:+.0f} | " +
                              " | ".join(f"{rate[k]:.3f}" for k in TESTS) + " |", flush=True)
    print("\nHighest false-positive rate over the 120 'none' settings: " +
          ", ".join(f"{k} {worst[k]:.3f}" for k in TESTS))
    print("Lowest power over the 120 'alarm' settings: " + ", ".join(f"{k} {weakest[k]:.3f}" for k in TESTS))
    print(f"# runtime {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
