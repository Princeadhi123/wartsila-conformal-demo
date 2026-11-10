---
id: KB-GL-008
title: When forecast uncertainty is unusually wide - guideline
type: ops_guideline
domain: energy
system: power_system
tags: [forecast uncertainty, wide interval, conservative dispatch]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Why intervals widen
Intervals widen around regime changes, sparse or stale input data, extreme weather in the forecast, or operating points far from the model's training conditions. Width is itself information: the model is honestly reporting that the outcome is less predictable than usual, which is exactly when conservative cover is cheapest relative to its value.

Check whether recent input data gaps or outages correlate with the widening before treating it as a genuine weather signal.

## Operational response
1. Size reserves to the upper bound even when the point forecast looks safe.
2. Prefer flexible, short-lead commitments over long-start units until the interval narrows.
3. Increase the monitoring cadence and re-run the forecast as fresh data arrives.
4. Verify the data feeds are healthy so the width is real rather than a pipeline fault.

5. Record the width history so post-event review can judge the response taken.

## What not to do
Do not dismiss an unusually wide interval as a model artefact without checking the inputs, and do not average it away or plan to the midpoint. Acting on the bound rather than the central estimate is exactly what calibrated uncertainty is for; a midpoint plan on a wide-interval day is the classic recipe for an avoidable shortfall event.

The cost of one cautious day is usually far below the cost of one shortfall event.
