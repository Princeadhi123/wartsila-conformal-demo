---
id: KB-GL-004
title: Handling watch versus confirmed alerts - guideline
type: ops_guideline
domain: marine
system: alerting
tags: [watch, confirmed, triage, interval]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## What a watch state means
The predictive interval straddles the action threshold, so the evidence is insufficient to act. A watch is not a fault and not a dismissal; it is a prompt to gather better evidence cheaply before committing resources. Most watches resolve either to confirmed decay or to instrumentation explanations, and both outcomes are cheaper when the watch is worked properly.

Typical watch duration is hours to days, until a better operating point or new data resolves it.

## Actions for a watch alert
1. Validate the relevant sensors before trusting the estimate.
2. Check the load point at which the alert arose; low load weakens the evidence.
3. Schedule a confirmation run at a higher steady load point if observability was poor.
4. Tighten the monitoring cadence until the state resolves.
5. Revisit the alert after each new steady-state measurement rather than letting it age.

## Actions for a confirmed alert
The entire interval lies beyond the threshold, so proceed to the relevant procedure: wash, borescope inspection, de-rate or dispatch as applicable. Record the decision together with the interval bounds and coverage level so later reviews can audit whether the action was justified. Where the action commits off-hire time or parts, escalate per the escalation matrix before executing.

A confirmed alert later disproven by inspection should be fed back as a false-positive case.
