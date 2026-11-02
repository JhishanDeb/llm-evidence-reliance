"""Synthetic PhysioNet 2012 data in the real format, for the offline tests only: folders set-a and set-b with one
'Time,Parameter,Value' file per ICU stay, and Outcomes-a.txt / Outcomes-b.txt with the column In-hospital_death.
usage: python make_fake_p12.py OUT_DIR N_PER_SET"""
import sys
from pathlib import Path

import numpy as np

out, n_per_set = Path(sys.argv[1]) / "physionet2012", int(sys.argv[2])
# parameter -> (mean, sd, direction with severity, share of stays where it is measured)
SERIES = {"HR": (87, 15, 1, 1.0), "GCS": (11, 3, -1, 0.98), "BUN": (27, 15, 1, 0.98), "Creatinine": (1.5, 0.8, 1, 0.98),
          "Glucose": (140, 40, 1, 0.97), "HCO3": (23, 4, -1, 0.97), "HCT": (31, 5, -1, 0.98), "K": (4.1, 0.6, 1, 0.97),
          "Mg": (2.0, 0.3, 0, 0.96), "Na": (139, 4, 0, 0.97), "Platelets": (190, 80, -1, 0.97), "Temp": (37.0, 0.7, 1, 0.99),
          "WBC": (12, 5, 1, 0.97), "Urine": (120, 60, -1, 0.97), "NISysABP": (118, 20, -1, 0.85),
          "NIDiasABP": (58, 12, -1, 0.85), "NIMAP": (77, 13, -1, 0.85), "RespRate": (19, 5, 1, 0.3),
          "Lactate": (2.5, 1.5, 1, 0.55), "pH": (7.38, 0.06, -1, 0.75), "FiO2": (0.5, 0.15, 1, 0.68),
          "PaO2": (140, 60, -1, 0.75), "PaCO2": (40, 8, 0, 0.75), "SysABP": (120, 20, -1, 0.7), "DiasABP": (59, 11, -1, 0.7),
          "MAP": (80, 13, -1, 0.7), "Bilirubin": (2.0, 3, 1, 0.42), "Albumin": (2.9, 0.6, -1, 0.45),
          "ALP": (115, 80, 1, 0.42), "ALT": (300, 500, 1, 0.43), "AST": (400, 700, 1, 0.43), "SaO2": (96, 3, -1, 0.45),
          "TroponinT": (1.0, 2.0, 1, 0.22), "Weight": (82, 22, 0, 0.92)}
rng = np.random.default_rng(12)
for letter, start in (("a", 132539), ("b", 142673)):
    d = out / f"set-{letter}"
    d.mkdir(parents=True, exist_ok=True)
    rows = ["RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"]
    for k in range(n_per_set):
        rid = start + k
        sev = rng.normal(0, 1)
        dead = int(rng.random() < 1 / (1 + np.exp(-(-2.0 + 1.1 * sev))))
        lines = ["Time,Parameter,Value", f"00:00,RecordID,{rid}", f"00:00,Age,{int(np.clip(rng.normal(64, 17), 16, 90))}",
                 f"00:00,Gender,{int(rng.random() < 0.56)}",
                 f"00:00,Height,{-1 if rng.random() < 0.47 else round(float(rng.normal(170, 10)), 1)}",
                 f"00:00,ICUType,{int(rng.integers(1, 5))}"]
        for p, (m, sd, dirn, share) in SERIES.items():
            if rng.random() > share:
                continue
            for _ in range(int(rng.integers(1, 6))):
                t = int(rng.integers(0, 48 * 60))
                v = m + dirn * 0.5 * sd * sev + sd * rng.normal()
                if p in ("FiO2",):
                    v = float(np.clip(v, 0.21, 1.0))
                elif p == "pH":
                    v = float(np.clip(v, 6.9, 7.7))
                elif p == "GCS":
                    v = float(np.clip(round(v), 3, 15))
                elif p == "SaO2":
                    v = float(np.clip(v, 70, 100))
                else:
                    v = float(abs(v))
                lines.append(f"{t // 60:02d}:{t % 60:02d},{p},{v:.2f}")
        if rng.random() < 0.6:
            lines.append("00:30,MechVent,1")
        (d / f"{rid}.txt").write_text("\n".join(lines) + "\n")
        rows.append(f"{rid},{int(10 + 3 * sev)},{int(5 + 2 * sev)},{int(rng.integers(2, 30))},-1,{dead}")
    (out / f"Outcomes-{letter}.txt").write_text("\n".join(rows) + "\n")
print("wrote", 2 * n_per_set, "ICU stays under", out)
