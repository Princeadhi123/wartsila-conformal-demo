---
id: KB-LOG-003
title: Reserve and dispatch operations logbook
type: maintenance_log
domain: energy
system: power_system
tags: [ops log, reserve shortfall, over-commitment, dispatch]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## 2025-04-22 - Evening ramp shortfall
Residual load ramped about 60 MW above the forecast upper bound for 35 minutes during the evening peak. Committed fast reserve had been sized to the point forecast, not the bound. BESS covered the first minutes and one genset started late, so import was used briefly. The post-event rule was adopted: always size to the upper bound of the calibrated interval.

## 2025-09-03 - Calm-day over-commitment
A wide interval driven by an unstable weather-model input led to committing two extra gensets that were barely loaded all day; fuel and start costs were incurred that hindsight showed unnecessary. The review added a step: before committing long-start units, check why the interval is wide and whether short-lead capacity would cover it more cheaply.

The cost breakdown was attached to the sizing guideline as a worked example.

## 2025-11-16 - Wide-interval day handled well
Ahead of a frontal passage the interval ran at about 2.4 times its typical width. BESS was pre-positioned and one fast-start unit committed against the upper bound while long-start units were deliberately held back. Realised residual load tracked inside the interval all day: no shortfall and modest incremental cost, which was accepted as the price of the cover.

The same playbook was reused two weeks later with a similar result.
