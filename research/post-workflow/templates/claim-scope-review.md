# Claim scope review

State the narrowest claim supported by the retained evidence, its baseline,
population, runtime, hardware and exclusions. Record the independent reviewer's
name and actual review result. Do not turn this template into a self-issued
acceptance receipt.

Read the title, opening, headings, image text and alt text, audience variants,
X and LinkedIn copy against that claim. Check these failure modes:

- A distribution or average described as a guarantee about each output.
- An "always", "only", "any" or "never" claim with a known exception.
- A measured gap that does not compare the mechanism being claimed.
- A control that cannot contradict the claim, or lacks independent seeds.
- A guarantee missing implementation conditions, preprocessing or masks.
- A claimed gain without its material trade-offs: added costs or requirements,
  included versus excluded work, losing or untested cases, and what is given up.
  Distinguish measured costs from possible risks. Do not invent a downside.
- A framework example naming the wrong runner, backend, kernel or version.
- A ratio without its actual baseline and measured population.
- Commands that fail from a clean checkout or silently skip required GPU work.
- A compute or link floor using the wrong instruction rate or bandwidth direction.
- Byte counts omitting metadata, scales or replicas, or assuming unattainable bandwidth.
- Scaling results that mix fixed total load with fixed load per GPU, or omit either per-GPU or total results.
- A fitted or subtracted component presented as directly measured, without checking all available points and residuals.
- A decision rule whose own numbers do not support its stated recommendation or objective.
- Communication counts that conflate one copy per pick with one per destination or omit the local share.
- Speculative-decoding steps conflated with tokens checked or emitted.
- An occupancy capacity bound conflated with concurrent resident slots or
  distinct participants, without the residency assumption.
- Fewer-started-block counters presented as proof of a scheduling policy
  without checking cancellation or work stealing.

Explain what the reader will learn or decide before introducing the mechanism.
Carry one introduced example through difficult steps and say where the analogy
stops matching. Use plain audience labels. Keep unmeasured and causal claims
explicitly bounded, and retain failed controls.
