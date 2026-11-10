---
id: KB-GL-007
title: Handling wind ramp events - guideline
type: ops_guideline
domain: energy
system: power_system
tags: [wind ramp, forecast interval, dispatch, reserve]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Recognising ramp risk
Large forecast ramps, such as frontal passages or wind cut-off events, widen residual-load intervals and can create fast reserve needs. Watch the interval width and the ramp-rate forecast, not only the forecast level: a modest level with a steep upper-bound ramp can be the more dangerous case, especially in the evening demand peak.

Combine the weather outlook with the interval width: rising wind plus a widening interval is a pre-alert in practice.

## Response
Pre-start enough fast capacity to cover the upper-bound ramp over the event window, keep BESS state of charge high and, where the market allows, shift flexible load away from the ramp. During the event, dispatch per the hybrid guideline and monitor system frequency rather than chasing the point forecast minute by minute.

Tell trading and dispatch partners early when extra cover is being committed so balancing actions are coordinated.

## Post-event review
Compare the actual ramp against the forecast bounds after the event closes. If the realised residual load breached the upper bound, flag the calibration for review; repeated breaches in the same weather regime mean the intervals are too narrow there, and the sizing rule should be widened temporarily until the model is re-checked.

Record which assets responded and how quickly, since real ramp performance often differs from nameplate expectations.
