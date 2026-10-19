"""Simulation behind Deviation D1 in PREREGISTRATION.md.

Question: does an asymmetry measure stay at zero when the model treats
alarm-pointing and safe-pointing evidence symmetrically, whatever its
baseline lean, and does it turn positive when alarm-pointing evidence
really moves the model more?

Each simulated patient has a true outcome y, an XGBoost log-odds z, and a
baseline log-odds of the alarming outcome for the language model, mu, which
depends partly on z and partly on the model's own reading of the patient.
The model's no-evidence answer (C0) is read as l0 = mu + eta + noise, where
eta is an idiosyncratic patient-level difference that only the no-evidence
prompt has. Each patient is shown one alarm-pointing card and one
safe-pointing card (C2 and C4 in the study). One of them is the true card,
which points to alarm more often when z is high, as in the real data.

Four ways a model could respond to a card are simulated:
  add       the card adds a fixed amount to the log-odds
  adopt     the output moves part of the way toward the card's side
  switch    with some probability the model switches to the card's side
  saturate  a fixed push that shrinks when the model already leans that way

Each response is run with no asymmetry and with alarm-pointing cards 1.5
times as strong as safe-pointing cards; with a lean toward safe (mean
baseline -3) or toward alarm (+3); with true cards followed as much as, or
50% more than, flipped cards (truth 0 or 0.5); and with or without the
idiosyncratic no-evidence difference (eta SD 0 or 2).

Measures:
  delta        A - S, the registered primary measure (Section 6)
  logodds      mean shift under the alarm card minus mean shift under the
               safe card, in log-odds (the registered secondary measure)
  l0-adjusted  intercept of d = (la - l0) + (ls - l0) regressed on l0,
               (l0/s)^3 and the true-card direction t
  adjusted     the new primary statistic: the same regression, but on the
               baseline b predicted from z, z^2, z^3 and y (first stage),
               so that idiosyncrasies of the no-evidence reading cannot
               cause regression to the mean

Run:  python analysis/simulate_asymmetry_measures.py
Runtime: under a minute. The output is deterministic (seed 2026).
"""
import numpy as np

N_PATIENTS = 600
N_REPS = 300
SEED = 2026


def _intercept(x, d, t):
    s = x.std()
    if not np.isfinite(s) or s == 0:
        return float("nan")
    X = np.column_stack([np.ones_like(x), x, (x / s) ** 3, t])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(X, d, rcond=None)[0][0])


def predicted_baseline(l0, z, y):
    """First stage: OLS of l0 on [1, z, z^2, z^3, y] (z standardized); returns the fitted values."""
    zs = (z - z.mean()) / z.std()
    Z = np.column_stack([np.ones_like(zs), zs, zs ** 2, zs ** 3, y])
    return Z @ np.linalg.lstsq(Z, l0, rcond=None)[0]


def adjusted_asymmetry(l0, la, ls, t, z, y):
    """The new primary statistic. Same code as adjusted_fit in ler_main.py."""
    l0, la, ls, t, z, y = (np.asarray(v, float) for v in (l0, la, ls, t, z, y))
    if len(l0) < 10 or not np.isfinite(z.std()) or z.std() == 0:
        return float("nan")
    b = predicted_baseline(l0, z, y)
    return _intercept(b, (la - l0) + (ls - l0), t)


def simulate(rng, response, lean, alarm_factor, truth, eta_sd, n=N_PATIENTS):
    y = (np.arange(n) % 2).astype(float)
    z = 1.2 * (2 * y - 1) + rng.normal(0, 1.3, n)                       # XGBoost log-odds
    mu = lean + 3 * z + rng.normal(0, 4, n)                               # model's baseline log-odds
    t = np.where(rng.random(n) < 1 / (1 + np.exp(-1.5 * (z - 1))), 1.0, -1.0)   # true card direction
    l0 = mu + rng.normal(0, eta_sd, n) + rng.normal(0, 1, n)

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

    la, ls = respond(np.ones(n)), respond(-np.ones(n))
    c0_alarm, a_alarm, s_alarm = l0 > 0, la > 0, ls > 0
    delta = a_alarm[~c0_alarm].mean() - (~s_alarm[c0_alarm]).mean()
    logodds = (la - l0).mean() - (l0 - ls).mean()
    l0_adjusted = _intercept(l0, (la - l0) + (ls - l0), t)
    return delta, logodds, l0_adjusted, adjusted_asymmetry(l0, la, ls, t, z, y)


def main():
    rng = np.random.default_rng(SEED)
    print(f"{N_REPS} simulated studies of {N_PATIENTS} patients each; mean (SD) of each measure.")
    print("Rows with asymmetry 'none' should give 0; rows with 'alarm' should give a value above 0.\n")
    print("| response | asymmetry | truth | eta SD | lean | delta = A - S | logodds | l0-adjusted | adjusted |")
    print("|---|---|---|---|---|---|---|---|---|")
    for response in ("add", "adopt", "switch", "saturate"):
        for asym, factor in (("none", 1.0), ("alarm", 1.5)):
            for truth in (0.0, 0.5):
                for eta_sd in (0.0, 2.0):
                    for lean in (-3.0, 3.0):
                        r = np.array([simulate(rng, response, lean, factor, truth, eta_sd) for _ in range(N_REPS)])
                        cells = " | ".join(f"{r[:, j].mean():+.2f} ({r[:, j].std():.2f})" for j in range(4))
                        print(f"| {response} | {asym} | {truth:.1f} | {eta_sd:.0f} | {lean:+.0f} | {cells} |")


if __name__ == "__main__":
    main()
