# Experimental Analysis

- Proposed policy average total cost: 0.1514
- Baseline average total cost: 0.2798
- Cost reduction: 0.1285 (45.91%)
- Proposed coverage: 0.990, escalation: 0.001, diagnostic: 0.009
- Baseline coverage: 0.798, escalation: 0.202
- Proposed Delta EO (FNR): 0.0021
- Baseline Delta EO (FNR): 0.0434

Interpretation:
The adaptive routing policy learns when to allocate human effort under cost constraints while
the fairness guardrail explicitly penalizes AI autonomy for the disadvantaged group when FNR gaps widen.
This often trades some coverage for better subgroup parity and lower deployment risk.