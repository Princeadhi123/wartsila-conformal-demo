---
id: KB-PROC-006
title: Fast-start gas engine genset start and dispatch - procedure
type: procedure
domain: energy
system: gas_engine_genset
tags: [genset, fast start, dispatch, reserve]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Dispatch decision
Start a fast-start gas engine genset when a reserve shortfall risk is confirmed, when the committed-minus-upper-bound margin turns negative, or to cover a predicted ramp that exceeds committed flexibility. Prefer the fewest starts that cover the upper bound; each start has a cost and a small failure probability that should not be spent casually.

Check that the start does not duplicate cover already available from BESS or imports.

If the risk is only a watch, place the likely units on standby and confirm their availability and synchronisation readiness without issuing the start.

## Start sequence
1. Confirm unit availability, fuel supply and synchronisation readiness.
2. Issue the start command; the unit should reach rated speed and synchronise within its certified start time.
3. Load the unit to the ordered MW ramp and monitor exhaust temperatures and vibration.
4. Confirm the reserve margin returns non-negative and log the dispatch.

5. Record the start attempt as successful or failed in the reliability log.

## Handback
When the risk window passes, unload in controlled steps, run the recommended cool-down and return the unit to standby rather than tripping it hot. Log the actual start time and loading profile; these records feed start-reliability statistics that refine future dispatch decisions and maintenance intervals.

A hot trip is reserved for genuine emergencies only, since repeated hot shutdowns shorten engine life and invalidate start-time statistics.
