---
id: KB-GL-003
title: Low-load observability and the confirmation run - guideline
type: ops_guideline
domain: marine
system: observability
tags: [low load, observability, confirmation run, interval width]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Why low load reduces observability
At low ship speed and low lever position the gas turbine operates far from its design point, so the measurements carry little information about kMc and kMt and the prediction intervals widen. Sensor noise also becomes large relative to the decay signal at part load. An alert raised at low load therefore rests on intrinsically weaker evidence than the same alert at cruise.

## Confirmation run procedure
1. When sea state and schedule allow, bring the plant to a steady higher load point, typically at least 70 percent of the alert-time power or a standard reference speed.
2. Hold the point steady for at least 10 to 15 minutes so the estimate can settle.
3. Re-estimate kMc and kMt and compare the new interval against the threshold.

## Interpreting the result
If the confirmation interval lies entirely beyond the threshold, treat the alert as confirmed and act. If it still straddles, keep watching and repeat the run at the next opportunity; repeated steady points also tighten the estimate. Do not schedule invasive maintenance on a low-load indication alone: a watch at low load is a request for a confirmation run, not a work order.
