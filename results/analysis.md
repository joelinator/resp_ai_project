# Experimental Analysis

- Proposed policy average total cost: 0.2221
- Baseline average total cost: 0.3837
- Cost reduction: 0.1616 (42.12%)
- Proposed coverage: 0.999, escalation: 0.000, diagnostic: 0.001
- Baseline coverage: 0.798, escalation: 0.202
- Proposed Delta EO (FNR): 0.0035
- Baseline Delta EO (FNR): 0.0554

Interpretation:
The adaptive routing policy learns when to allocate human effort under cost constraints while
the fairness guardrail explicitly penalizes AI autonomy for the disadvantaged group when FNR gaps widen.
This often trades some coverage for better subgroup parity and lower deployment risk.