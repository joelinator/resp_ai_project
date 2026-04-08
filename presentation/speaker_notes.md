# Speaker Notes (5-slide deck)

## Slide 1 — Title
- Introduce the problem: human-AI collaboration for personalized exercise recommendation.
- Emphasize that this is not only a prediction problem; it is a routing and resource-allocation problem.
- Preview: task framing, simulation setup, workflow, and empirical results.

## Slide 2 — Task and Importance
- Explain the three actions clearly:
  - `A`: AI makes fit/not-fit judgment autonomously
  - `B`: escalate fit/not-fit judgment to human teacher
  - `C`: request diagnostic micro-quiz
- Explain objective: minimize expected wrong-answer loss plus action deployment cost.
- Clarify task goal explicitly: estimate if the student is proficient on the skill (needs support vs does not need support).
- Clarify reframed loss:
  - if judged fit: loss is `1` only when student fails,
  - if judged not fit: apply imputed opportunity loss `0.4`.
- Motivate importance:
  - educational harm from wrong recommendations,
  - limited teacher capacity,
  - fairness risk if one group is systematically routed to lower-quality decisions.

## Slide 3 — Simulation and Policy Setup
- Input features:
  - rolling student-skill correctness history,
  - attempt count,
  - recency,
  - skill identity.
- Explain calibration and uncertainty:
  - probabilities are calibrated with a sigmoid mapping,
  - confidence is `max(p, 1-p)`,
  - uncertainty is `1 - confidence`.
- Explain teacher simulation:
  - base competence with gender/skill effects and Gaussian noise.
- Mention deployment cost settings: `A:+0.0`, `B:+0.9`, `C:+0.5`.

## Slide 4 — Baseline vs Proposed Workflow
- Baseline:
  - single threshold over confidence,
  - can only choose AI or human escalation.
- Proposed:
  - contextual LinUCB selects among three actions,
  - learns from observed total cost,
  - fairness guardrail monitors EO gap and adjusts AI-penalty if needed.
- Clarify fairness target:
  - FNR is computed on students needing support (`y=0`),
  - false negative means the chosen decision-maker (AI/Human/Diagnostic) judges `fit` when student is actually not proficient.
- Point at the flowchart to show where fairness updates happen in the loop.

## Slide 5 — Results and Takeaways
- Read the key numbers:
  - cost: `0.384 -> 0.222` (~42.1% reduction),
  - AI coverage: `0.798 -> 0.9989`,
  - EO gap: `0.0554 -> 0.0035`.
- Mention that this is from the 5x-timestep simulation stream (`180,000` total simulated decisions).
  - Source files: `results_reframed_proficiency_5x/metrics_summary.csv` and `results_reframed_proficiency_5x/risk_coverage_overall.png`.
- Acknowledge trade-off:
  - average loss increases (`0.202 -> 0.222`),
  - absolute FNR remains high despite better parity between groups.
- Final message:
  - optimizing deployment utility and fairness can differ from optimizing raw predictive accuracy.
  - this is exactly why a routing policy is needed around the base model.
