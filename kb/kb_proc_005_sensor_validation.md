---
id: KB-PROC-005
title: Sensor validation before acting on alerts - procedure
type: procedure
domain: marine
system: sensors
tags: [sensor validation, T48, thermocouple, pressure transmitter, drift]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## T48 thermocouple spread check
Compare all T48 probes at a steady load point: the spread should normally stay within a few tens of degrees. One probe drifting apart from the pack indicates a failing thermocouple rather than real turbine decay. Exclude the suspect channel, re-estimate the health coefficient and only then decide whether the alert stands. Log which channel was excluded so later trending stays consistent.

## Pressure transmitter drift check
Compare P2 and P48 transmitter readings against local gauges or redundant channels at a known steady condition. Zero or span drift produces false decay estimates in either direction, so check both bias and slope where possible. A reading that jumps after a maintenance visit or a vibration event is a classic drift signature and should be recalibrated before trusting it.

## General validation rules
Check for step changes coinciding with maintenance events, and verify the indication at a second load point before acting. Never act on a decay alert that rests on a single suspect channel; a validated alert should survive re-estimation with the suspect input removed. If removing one channel flips the alert state, the alert is instrumentation-limited, not a plant condition.

Suspect channels are replaced or recalibrated at the next stop, then re-trended.
