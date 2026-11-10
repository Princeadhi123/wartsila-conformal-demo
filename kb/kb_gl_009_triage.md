---
id: KB-GL-009
title: Alert triage under high alert load - remote operations centre
type: ops_guideline
domain: ops
system: operations_centre
tags: [triage, alert load, remote operations, prioritisation]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Triage order
Under high alert load, handle confirmed alerts before watch alerts, safety- and supply-affecting alerts before efficiency-affecting ones, and time-critical alerts before deferrable ones. An uncertain watch alert never outranks a confirmed one; its correct handling is usually cheaper evidence-gathering anyway, which can often be delegated or scheduled.

Write the current order on the shift board so the next operator inherits the reasoning, not just the list.

## Batching and deferral
Group alerts that share an asset or a driver into a single incident so the operator sees one coherent problem rather than five fragments. Defer purely informational alerts and set explicit revisit times for deferred items; nothing should linger unowned. Re-review the deferred queue at every shift handover and after the active queue drains.

Suppression rules for known benign repeats belong in the same review.

## Workload signals
Watch the operator's own cognitive load: many active alerts plus a burst of new ones degrades decision quality. Under high load, shorten advisories, surface only the top actions and escalate the triage burden itself when the load stays high. A quiet period should be used to clear the deferred queue, not to generate new work.

Fatigue is itself an operational risk and belongs in the shift report.
