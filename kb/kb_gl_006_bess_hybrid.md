---
id: KB-GL-006
title: BESS plus engine hybrid dispatch - guideline
type: ops_guideline
domain: energy
system: hybrid_power
tags: [BESS, battery, hybrid dispatch, genset, ramp]
---
> Illustrative synthetic document written for this demo from general, publicly known engineering practice. Not Wärtsilä documentation.

## Role split
Use the battery energy storage system for fast, short-duration cover: ramps, the first minutes of a shortfall, and forecast-error absorption. Use gas engine gensets for sustained energy over tens of minutes to hours, where fuel per megawatt-hour is lower. Do not spend battery energy on sustained deficits that engines can carry more cheaply.

This split also matches the assets' wear costs: frequent starts age engines, while deep cycling ages batteries.

## Coordination
When the committed-minus-upper-bound margin is negative, pre-position BESS reserve for the fast component and start engines so their available output covers the sustained component. Split the net requirement, defined as upper bound minus committed capacity, by expected duration rather than committing everything at once. Re-check the split whenever the forecast updates.

Communicate the split explicitly so operators know which asset answers first and which answers longest.

## State-of-charge management
Preserve state of charge ahead of forecast ramp windows and recharge during low-risk periods when the interval is narrow and margin is comfortably positive. If SOC is low and forecast uncertainty is wide, prefer engine cover so the battery remains available for genuinely fast events that engines cannot reach in time.

A battery held at mid-charge all day is wasted flexibility; schedule its energy deliberately around the forecast risk windows.
