"""Synthetic UCI Diabetes 130 data in the real format ('?' = missing, IDS_mapping.csv with 3 stacked tables)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out, n = Path(sys.argv[1]) / "diabetes130", int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5)
DRUGS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
         "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
         "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
         "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
pat = rng.integers(1, int(n * 0.75), n)
risk = rng.normal(0, 1, n)
df = pd.DataFrame({
    "encounter_id": np.sort(rng.choice(np.arange(10000, 10000 + 50 * n), n, replace=False)),
    "patient_nbr": pat,
    "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian", "?"], n, p=[.7, .18, .02, .02, .01, .07]),
    "gender": rng.choice(["Female", "Male", "Unknown/Invalid"], n, p=[.53, .4699, .0001]),
    "age": rng.choice([f"[{a}-{a + 10})" for a in range(0, 100, 10)], n),
    "weight": np.where(rng.random(n) < 0.97, "?", rng.choice(["[75-100)", "[50-75)", "[100-125)"], n)),
    "admission_type_id": rng.choice([1, 2, 3, 5, 6, 8], n),
    "discharge_disposition_id": rng.choice([1, 2, 3, 6, 11, 13, 14, 18, 19, 20, 22, 25], n,
                                           p=[.55, .03, .12, .12, .02, .01, .01, .04, .01, .01, .04, .04]),
    "admission_source_id": rng.choice([1, 2, 4, 6, 7, 17], n),
    "time_in_hospital": rng.integers(1, 15, n),
    "payer_code": rng.choice(["?", "MC", "HM", "SP", "BC"], n),
    "medical_specialty": rng.choice(["?", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology"], n),
    "num_lab_procedures": rng.integers(1, 100, n),
    "num_procedures": rng.integers(0, 7, n),
    "num_medications": rng.integers(1, 60, n),
    "number_outpatient": rng.poisson(0.4, n),
    "number_emergency": rng.poisson(0.2, n),
    "number_inpatient": np.clip(rng.poisson(0.6, n) + (risk > 1).astype(int), 0, None),
    "diag_1": rng.choice(["428", "414", "786", "410", "486", "427", "250.83", "V57", "?"], n),
    "diag_2": rng.choice(["276", "428", "250", "427", "401", "?"], n),
    "diag_3": rng.choice(["250", "401", "276", "428", "?"], n),
    "number_diagnoses": rng.integers(1, 17, n),
    "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n, p=[.94, .02, .02, .02]),
    "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], n, p=[.82, .04, .08, .06]),
})
for dname in DRUGS:
    df[dname] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[.8, .15, .03, .02])
df["change"] = rng.choice(["No", "Ch"], n)
df["diabetesMed"] = rng.choice(["Yes", "No"], n, p=[.77, .23])
p30 = 1 / (1 + np.exp(-(-2.3 + 0.9 * risk + 0.3 * df["number_inpatient"])))
u = rng.random(n)
df["readmitted"] = np.where(u < p30, "<30", np.where(u < p30 + 0.35, ">30", "NO"))
df.to_csv(out / "diabetic_data.csv", index=False)
(out / "IDS_mapping.csv").write_text("""admission_type_id,description
1,Emergency
2,Urgent
3,Elective
4,Newborn
5,Not Available
6,NULL
7,Trauma Center
8,Not Mapped
,
discharge_disposition_id,description
1,Discharged to home
2,Discharged/transferred to another short term hospital
3,Discharged/transferred to SNF
6,Discharged/transferred to home with home health service
11,Expired
13,Hospice / home
14,Hospice / medical facility
18,NULL
19,"Expired at home. Medicaid only, hospice."
20,"Expired in a medical facility. Medicaid only, hospice."
22,Discharged/transferred to another rehab fac including rehab units of a hospital .
25,Not Mapped
,
admission_source_id,description
1, Physician Referral
2,Clinic Referral
4,Transfer from a hospital
6, Transfer from another health care facility
7, Emergency Room
17,NULL
""")
print("wrote", n, "encounters to", out)
