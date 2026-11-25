# Key findings for presentation

- XGBoost: Raw + CCS had the highest five-fold mean AP (0.2323); adding CCS to raw ICD-9 changed AP by +0.0037 (patient bootstrap 95% CI [+0.0022, +0.0053]).
- HGB: Raw + CCS had the highest five-fold mean AP (0.2392); adding CCS to raw ICD-9 changed AP by +0.0019 (patient bootstrap 95% CI [+0.0005, +0.0034]).
- CatBoost: Raw + CCS had the highest five-fold mean AP (0.2469); adding CCS to raw ICD-9 changed AP by +0.0004 (patient bootstrap 95% CI [-0.0008, +0.0018]).

## Existing ensemble decision

The original CCS HGB + raw CatBoost ensemble has five-fold mean AP 0.247374; using raw+CCS CatBoost gives 0.247388. The paired AP difference is +0.000014 (95% CI [-0.000681, +0.000934]). This comparison does not establish an ensemble improvement.

## Interpretation

This experiment evaluates diagnosis representation within each model family. The earlier GPU classic baselines are historical references; the controlled classic comparison uses fresh CPU runs for all three representations. The saved CatBoost variants are reused because their feature construction, selection grid and outer folds match.
