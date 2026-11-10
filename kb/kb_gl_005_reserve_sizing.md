---
id: KB-GL-005
title: Reserve sizing against residual-load forecast uncertainty
type: ops_guideline
domain: energy
system: power_system
tags: [reserve sizing, forecast interval, upper bound, spinning reserve]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Sizing principle
Hold spinning or fast-start reserve at least equal to the upper bound of the calibrated residual-load forecast interval for the commitment horizon, plus the largest single contingency where the grid code requires it. The upper bound, not the point forecast, is the sizing target: a point forecast carries no probability statement about shortfalls and will be exceeded roughly half the time.

## Working with the upper bound
If committed reserve falls short of the upper bound, either commit additional fast capacity or accept a quantified shortfall risk with explicit sign-off from the duty manager. Track margin continuously as committed capacity minus the upper bound; negative margin is an alert condition, and a margin near zero during a ramp window deserves a watch even before it goes negative.

## Documentation
Record the forecast interval, its coverage level, the committed reserve and the margin at each decision gate, together with the reason for any accepted shortfall. These records make later reviews of over- and under-commitment possible and are the raw material for improving the sizing rule itself.

Review these records monthly against realised outcomes to keep the commitment discipline honest and to spot systematic bias early.
