"""Synthetic PhysioNet 2019 data in the real format: one pipe-separated .psv per patient, 'NaN' = missing,
SepsisLabel = 1 from 6 h before onset. Offline tests only."""
import sys
from pathlib import Path
import numpy as np

out, n_per_set = Path(sys.argv[1]), int(sys.argv[2])
COLS = ("HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|"
        "Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|"
        "TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel").split("|")
MEAN = {"HR": 85, "O2Sat": 97, "Temp": 36.9, "SBP": 122, "MAP": 82, "DBP": 62, "Resp": 18, "EtCO2": 33,
        "BaseExcess": 0, "HCO3": 24, "FiO2": 0.5, "pH": 7.38, "PaCO2": 41, "SaO2": 93, "AST": 200, "BUN": 23,
        "Alkalinephos": 100, "Calcium": 8, "Chloride": 105, "Creatinine": 1.5, "Bilirubin_direct": 1.8,
        "Glucose": 135, "Lactate": 2.6, "Magnesium": 2.0, "Phosphate": 3.5, "Potassium": 4.1, "Bilirubin_total": 1.5,
        "TroponinI": 8, "Hct": 31, "Hgb": 10.5, "PTT": 41, "WBC": 11, "Fibrinogen": 287, "Platelets": 196}
VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"}
rng = np.random.default_rng(11)
for s, start in (("training_setA", 1), ("training_setB", 100001)):
    d = out / "physionet2019" / s
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n_per_set):
        L = int(rng.integers(12, 90))
        septic = rng.random() < 0.18
        onset = int(rng.integers(8, L + 1)) if septic else None
        sev = rng.normal(0, 1) + (1.0 if septic else 0)
        age, gender = float(np.clip(rng.normal(62, 16), 18, 99)), int(rng.random() < 0.55)
        hosp = round(float(-abs(rng.normal(0, 30))), 2)
        lines = ["|".join(COLS)]
        for h in range(1, L + 1):
            row = []
            for c in COLS[:34]:
                p_obs = 0.85 if c in VITALS else 0.06
                if rng.random() > p_obs:
                    row.append("NaN")
                    continue
                m = MEAN[c] * (1 + 0.08 * sev * (1 if c in ("HR", "Resp", "Lactate", "WBC", "Temp") else -0.3))
                row.append(f"{m * (1 + 0.1 * rng.normal()):.2f}")
            lab = 1 if (septic and h >= onset - 6) else 0
            row += [f"{age:.2f}", str(gender), "NaN" if rng.random() < 0.3 else str(int(rng.random() < 0.5)),
                    "NaN", f"{hosp:.2f}", str(h), str(lab)]
            lines.append("|".join(row))
        (d / f"p{start + k:06d}.psv").write_text("\n".join(lines) + "\n")
print("wrote", 2 * n_per_set, "patients under", out / "physionet2019")
