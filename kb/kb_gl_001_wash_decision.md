---
id: KB-GL-001
title: Compressor wash decision criteria versus fuel penalty
type: ops_guideline
domain: marine
system: gas_turbine_compressor
tags: [decision criteria, fuel penalty, water wash, kMc]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Decision criteria
Act when confirmed kMc decay exceeds the action threshold, typically about 0.025 below the healthy baseline, and the estimated fuel penalty exceeds the cost and disruption of a wash. Confirmed means the calibrated interval, not just the point estimate, clears the threshold. An interval straddling the threshold is a watch, so gather more evidence or run a confirmation point first.

## Fuel-penalty trade-off
Each 0.01 of kMc decay costs roughly 0.7-1.2 percent fuel at cruise depending on the plant. Estimate the penalty from measured fuel flow at fixed ship speed versus the healthy baseline, and weigh it against wash cost, off-hire time and the operating schedule. Prefer an online wash for early decay with kMc above about 0.975, and an offline wash once decay is established or after two online washes with poor recovery.

## Worked example
kMc estimated 0.968 at 15 kn with fuel flow about 2.1 percent above baseline: the penalty outweighs one day of reduced availability, so schedule an offline wash at the next port call that offers enough cooling time. Re-check the interval after the confirmation run if the alert arose at low load, and record the fuel-penalty estimate alongside the work order for later audit.
