# Dev-phase report

Model: Qwen/Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28, instruct), backend vllm
Protocol: prereg-v1, sha256 8457d962d514885f76cdec09724aaa9d41c3e6e237b1c163c44f2e3e49cfc4f9

## Data

- **physionet2012**: {"records_parsed": 8000, "with_outcome": 8000, "positives": 1122, "licence": "Open Data Commons Attribution License v1.0", "seconds": 253.6}
  - split {"train": 4800, "dev_pool": 773, "test_pool": 2342, "pilot_removed_dev": 27, "pilot_removed_test": 58, "train_positives": 673}
  - XGBoost AUROC on the dev pool (16 features): 0.798
  - test-pool capacity {"dev_pool": {"eligible": 758, "eligible_pos": 95, "eligible_neg": 663}, "test_pool": {"eligible": 2299, "eligible_pos": 302, "eligible_neg": 1997}, "test_per_class_planned": 300, "test_fallback_rule_applies": false}
  - cards {"card_points_alarm_share": 0.4, "card_agrees_with_xgb_label": 0.97, "partners_missing": 0}
- **physionet2019**: {"files": 40336, "status_excl_empty": 15, "status_excl_label_le24": 1351, "status_excl_short": 9017, "status_neg": 28373, "status_pos": 1580, "included": 29953, "positives": 1580, "licence": "Creative Commons Attribution 4.0 International (CC BY 4.0)", "seconds": 190.5}
  - split {"train": 17971, "dev_pool": 2995, "test_pool": 8987, "pilot_removed_dev": 0, "pilot_removed_test": 0, "train_positives": 948}
  - XGBoost AUROC on the dev pool (16 features): 0.730
  - test-pool capacity {"dev_pool": {"eligible": 2833, "eligible_pos": 155, "eligible_neg": 2678}, "test_pool": {"eligible": 8527, "eligible_pos": 466, "eligible_neg": 8061}, "test_per_class_planned": 300, "test_fallback_rule_applies": false}
  - cards {"card_points_alarm_share": 0.32, "card_agrees_with_xgb_label": 0.83, "partners_missing": 0}
- **diabetes130**: {"encounters": 101766, "first_encounters": 71518, "excluded_death_or_hospice": 1545, "included": 69973, "positives": 6277, "id_mapping_tables": {"admission_type_id": 8, "discharge_disposition_id": 30, "admission_source_id": 25}, "licence": "Creative Commons Attribution 4.0 International (CC BY 4.0)", "seconds": 3.4}
  - split {"train": 41983, "dev_pool": 6997, "test_pool": 20993, "pilot_removed_dev": 0, "pilot_removed_test": 0, "train_positives": 3766}
  - XGBoost AUROC on the dev pool (16 features): 0.597
  - test-pool capacity {"dev_pool": {"eligible": 6997, "eligible_pos": 628, "eligible_neg": 6369}, "test_pool": {"eligible": 20993, "eligible_pos": 1883, "eligible_neg": 19110}, "test_per_class_planned": 300, "test_fallback_rule_applies": false}
  - cards {"card_points_alarm_share": 0.23, "card_agrees_with_xgb_label": 0.91, "partners_missing": 0}

## Validity (dev, early warning)

```
      dataset  n_prompts  yn_mass_median_alarm_A  yn_mass_min  n_yn_mass_below_0.01  n_invalid_zero_mass  C0_accuracy  C0_share_alarm  valid_yn_mass  valid_C0_acc  tokens_median  tokens_max
physionet2012       1400                   0.999        0.773                     0                    0         0.71            0.29           True          True           1646        1672
physionet2019       1400                   0.992        0.628                     0                    0         0.66            0.50           True          True           2195        2238
  diabetes130       1400                   0.993        0.570                     0                    0         0.54            0.24           True         False           1232        1260
```

## Runtime projection

0.199 s per prompt (vllm)

- planned (300 per class, or fallback): 25200 prompts per model, 1.68 h per model, 6.72 h for 4 models (budget 30 h)
- compute fallback (200 per class): 16800 prompts per model, 1.21 h per model, 4.86 h for 4 models (budget 30 h)

## To record under Deviations before the main run

- Eligibility operationalised as at least 10 of 16 selected features measured (>= 60%); the pilot code used int(0.6 x 16) = 9.
- Diabetes 130: the first encounter per patient is taken first, then excluded if it ended in death or hospice (so such patients are dropped, not replaced by a later encounter).
- Validity check: median Yes+No mass over all conditions under the alarm framing with example set A; C0 accuracy under the alarm framing with example set A.
- Solved examples: drawn at random from all training-split patients with the registered seeds, shown in the order alarming, safe, alarming, safe.
- A prompt with no probability on any Yes/No token (Yes+No mass = 0) is counted and left out of the analysis; it is not read as 'No'.
- The Qwen/Qwen2.5-7B-Instruct chat template adds its own default system text when no system message is given: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'. It is used as is (no system message is passed).
- diabetes130: dev validity warning (median Yes+No mass 0.99, C0 accuracy 0.54); look at this before the main run.

## Exploratory dev results (not for decisions)

```
{
 "physionet2012": {
  "n_records": 1400,
  "n_invalid": 0,
  "by_condition": {
   "C0": {
    "n": 100,
    "accuracy": 0.71,
    "wilson": [
     0.6146110882442888,
     0.7898516425087772
    ],
    "auroc": 0.8488
   },
   "C2": {
    "n": 100,
    "accuracy": 0.71,
    "wilson": [
     0.6146110882442888,
     0.7898516425087772
    ],
    "auroc": 0.8456
   },
   "C4": {
    "n": 100,
    "accuracy": 0.6,
    "wilson": [
     0.5020025867910618,
     0.6905987135675411
    ],
    "auroc": 0.8
   },
   "C6": {
    "n": 100,
    "accuracy": 0.62,
    "wilson": [
     0.5220975529551107,
     0.7090240074752127
    ],
    "auroc": 0.7055999999999999
   }
  },
  "H1": {
   "acc_a": 0.71,
   "acc_b": 0.71,
   "diff": 0.0,
   "a_only_correct": 4,
   "b_only_correct": 4,
   "p": 1.0,
   "p_holm": 1.0,
   "supported": false
  },
  "H2": {
   "acc_a": 0.6,
   "acc_b": 0.71,
   "diff": -0.10999999999999999,
   "a_only_correct": 2,
   "b_only_correct": 13,
   "p": 0.00738525390625,
   "p_holm": 0.02215576171875,
   "supported": true
  },
  "H3": {
   "n": 100,
   "A": 0.07042253521126761,
   "S": 0.5517241379310345,
   "delta": -0.48130160271976685,
   "delta_ci": [
    -0.6751195480019493,
    -0.28817858954845255
   ],
   "p_boot": 0.0,
   "logodds_asym": -0.4289297218232275,
   "logodds_asym_ci": [
    -1.7387152507791537,
    0.8896399530221971
   ],
   "follow_alarm_card": 0.32,
   "follow_safe_card": 0.87,
   "flip_rate_C2_C4": 0.19,
   "p_holm": 0.0,
   "supported": false
  },
  "Q3_reversed": {
   "n": 100,
   "A": 0.06153846153846154,
   "S": 0.4857142857142857,
   "delta": -0.42417582417582417,
   "delta_ci": [
    -0.6016830294530154,
    -0.24421594457455117
   ],
   "p_boot": 0.0,
   "logodds_asym": -4.045398027973063,
   "logodds_asym_ci": [
    -5.748544327147438,
    -2.3861348871223718
   ],
   "follow_alarm_card": 0.33,
   "follow_safe_card": 0.8,
   "flip_rate_C2_C4": 0.21
  },
  "Q3_interpretation": "follows the answer word 'Yes'",
  "H5": {
   "n": 100,
   "overtrust": -0.2329261651078567,
   "ci": [
    -0.4534196921801346,
    -0.03212641151380266
   ],
   "acc_C4": 0.6,
   "mean_predicted_C4": 0.3670738348921433
  },
  "H6": {
   "n": 100,
   "auroc_with_flag": 0.5091145833333334,
   "auroc_without_flag": 0.48567708333333337,
   "diff": 0.0234375,
   "p_delong": 0.7470649666992091,
   "share_flagged": 0.19
  }
 },
 "physionet2019": {
  "n_records": 1400,
  "n_invalid": 0,
  "by_condition": {
   "C0": {
    "n": 100,
    "accuracy": 0.66,
    "wilson": [
     0.5627772885472462,
     0.7453847920265184
    ],
    "auroc": 0.6804
   },
   "C2": {
    "n": 100,
    "accuracy": 0.68,
    "wilson": [
     0.5833738138954762,
     0.7633085267500092
    ],
    "auroc": 0.7044
   },
   "C4": {
    "n": 100,
    "accuracy": 0.59,
    "wilson": [
     0.4920143005341078,
     0.6813268697886348
    ],
    "auroc": 0.5324
   },
   "C6": {
    "n": 100,
    "accuracy": 0.6,
    "wilson": [
     0.5020025867910618,
     0.6905987135675411
    ],
    "auroc": 0.6980000000000001
   }
  },
  "H1": {
   "acc_a": 0.68,
   "acc_b": 0.66,
   "diff": 0.020000000000000018,
   "a_only_correct": 7,
   "b_only_correct": 5,
   "p": 0.7744140625,
   "p_holm": 1.0,
   "supported": false
  },
  "H2": {
   "acc_a": 0.59,
   "acc_b": 0.66,
   "diff": -0.07000000000000006,
   "a_only_correct": 12,
   "b_only_correct": 19,
   "p": 0.28104151505976915,
   "p_holm": 0.5620830301195383,
   "supported": false
  },
  "H3": {
   "n": 100,
   "A": 0.46,
   "S": 0.36,
   "delta": 0.10000000000000003,
   "delta_ci": [
    -0.09878622288458351,
    0.28846153846153844
   ],
   "p_boot": 0.3262,
   "logodds_asym": -2.313882790984252,
   "logodds_asym_ci": [
    -4.102864823471633,
    -0.5365149243195207
   ],
   "follow_alarm_card": 0.71,
   "follow_safe_card": 0.68,
   "flip_rate_C2_C4": 0.43,
   "p_holm": 0.3262,
   "supported": false
  },
  "Q3_reversed": {
   "n": 100,
   "A": 0.7567567567567568,
   "S": 0.20634920634920634,
   "delta": 0.5504075504075505,
   "delta_ci": [
    0.375,
    0.7135416666666666
   ],
   "p_boot": 0.0,
   "logodds_asym": -0.4929035580714176,
   "logodds_asym_ci": [
    -2.045498395830411,
    1.0618835849880788
   ],
   "follow_alarm_card": 0.91,
   "follow_safe_card": 0.44,
   "flip_rate_C2_C4": 0.39
  },
  "Q3_interpretation": "follows the outcome (alarm aversion)",
  "H5": {
   "n": 100,
   "overtrust": 0.0606819961737084,
   "ci": [
    -0.08687403273767658,
    0.20459786753429152
   ],
   "acc_C4": 0.59,
   "mean_predicted_C4": 0.6506819961737084
  },
  "H6": {
   "n": 100,
   "auroc_with_flag": 0.6344349245206039,
   "auroc_without_flag": 0.5569155446756425,
   "diff": 0.07751937984496138,
   "p_delong": 0.1047787331648786,
   "share_flagged": 0.43
  }
 },
 "diabetes130": {
  "n_records": 1400,
  "n_invalid": 0,
  "by_condition": {
   "C0": {
    "n": 100,
    "accuracy": 0.54,
    "wilson": [
     0.44264860323368227,
     0.6343919169097589
    ],
    "auroc": 0.5818
   },
   "C2": {
    "n": 100,
    "accuracy": 0.59,
    "wilson": [
     0.4920143005341078,
     0.6813268697886348
    ],
    "auroc": 0.6784
   },
   "C4": {
    "n": 100,
    "accuracy": 0.48,
    "wilson": [
     0.384645517580519,
     0.5768342223477603
    ],
    "auroc": 0.476
   },
   "C6": {
    "n": 100,
    "accuracy": 0.54,
    "wilson": [
     0.44264860323368227,
     0.6343919169097589
    ],
    "auroc": 0.6200000000000001
   }
  },
  "H1": {
   "acc_a": 0.59,
   "acc_b": 0.54,
   "diff": 0.04999999999999993,
   "a_only_correct": 11,
   "b_only_correct": 6,
   "p": 0.332305908203125,
   "p_holm": 0.996917724609375,
   "supported": false
  },
  "H2": {
   "acc_a": 0.48,
   "acc_b": 0.54,
   "diff": -0.06000000000000005,
   "a_only_correct": 24,
   "b_only_correct": 30,
   "p": 0.49661743531793345,
   "p_holm": 0.5620830301195383,
   "supported": false
  },
  "H3": {
   "n": 100,
   "A": 0.6447368421052632,
   "S": 0.8333333333333334,
   "delta": -0.1885964912280702,
   "delta_ci": [
    -0.36759461178501657,
    0.0
   ],
   "p_boot": 0.0516,
   "logodds_asym": 0.03139371780315514,
   "logodds_asym_ci": [
    -1.291814363516001,
    1.4108048194511575
   ],
   "follow_alarm_card": 0.72,
   "follow_safe_card": 0.95,
   "flip_rate_C2_C4": 0.69,
   "p_holm": 0.1032,
   "supported": false
  },
  "Q3_reversed": {
   "n": 100,
   "A": 0.8333333333333334,
   "S": 0.40789473684210525,
   "delta": 0.4254385964912281,
   "delta_ci": [
    0.2328482328482328,
    0.6017009369301388
   ],
   "p_boot": 0.0006,
   "logodds_asym": -4.493215551743333,
   "logodds_asym_ci": [
    -5.391936181661205,
    -3.590521252488988
   ],
   "follow_alarm_card": 0.94,
   "follow_safe_card": 0.55,
   "flip_rate_C2_C4": 0.51
  },
  "Q3_interpretation": "follows the outcome (alarm aversion)",
  "H5": {
   "n": 100,
   "overtrust": 0.11641340581561266,
   "ci": [
    -0.0641451081369222,
    0.30391875130663437
   ],
   "acc_C4": 0.48,
   "mean_predicted_C4": 0.5964134058156126
  },
  "H6": {
   "n": 100,
   "auroc_with_flag": 0.47717171717171714,
   "auroc_without_flag": 0.46989898989899004,
   "diff": 0.0072727272727270975,
   "p_delong": 0.8235305974469596,
   "share_flagged": 0.69
  }
 }
}
```
