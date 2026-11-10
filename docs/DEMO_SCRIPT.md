# Meeting script — Wärtsilä Corporate Corner 1-on-1 (30 min)

Start the demo before the meeting: double-click `run_demo.bat`, open http://localhost:8501,
and click through every page once so everything is cached. Works fully offline.

All numbers below come from `artifacts/*_metrics.json` (seeded, reproducible with
`python scripts/run_pipeline.py`). If you re-run the pipeline, re-check them on the
"Trust & validation" page before quoting.

## Timing (from the organiser's guidelines)

| Slot | Minutes | What |
|---|---|---|
| Introductions | 2–5 | Who you are, TRILA / University of Turku, one line on your PhD |
| Presentation | ≤ 10 | 3 min framing + 5 min live demo + 2 min pilot ask |
| Discussion | 10–15 | Their needs — ask the open questions, listen |
| Next steps | 2–5 | Follow-up action, contact person, date |

## Presentation (10 min)

### 1. Framing — Overview page (2–3 min)

- "Operators cannot safely act on an AI that never says how sure it is. My research is about
  models that know — and state — their own uncertainty. I built a small end-to-end demo on
  public data to show what that looks like for marine and energy operations."
- Say this up front: **no Wärtsilä data, public datasets, synthetic manuals** — it is a
  mock-up of the idea, not a product. The marine data is a public gas-turbine frigate
  simulator; the method does not care which machine it is.
- Point at the four cards:
  - False alarms on held-out data: **5.0 % → 0.26 %** (about 19× fewer).
  - A 90 % interval that actually delivers **≈ 89 %** on unseen operating points.
  - Energy: an ordinary quantile model *claims* 90 % and delivers **77 %**; with adaptive
    conformal calibration it delivers **90.0 %**.
  - Every recommended action cites its source document.

### 2. Live demo — Marine console (3 min)

Open `http://localhost:8501/marine_console?hour=340`, speed 4 h/tick, press **Play**.

1. "The band is the 90 % interval. Watch it widen when the ship slows down — the model is
   telling us the health state is harder to observe at low load."
2. Around hour 347–361: the agent raises **WATCH** ("interval straddles the threshold — gather
   evidence"), while the conventional alarm fires. Toggle **Reveal ground truth**: the
   conventional alarms at hours 361 and 387 were **false**; the compressor only truly crossed
   the threshold at hour 404.
3. Hour 432: **CONFIRMED** — the whole interval is below the threshold. Open the advisory:
   the confidence statement, the reasons, the cited steps, the similar past case.
4. Drag "Simulated additional concurrent alerts" to 5 → the advisory collapses to **Focus**
   mode (two actions). "Same evidence, less text when the operator is busy."
5. "Jump to wash": after the wash at hour 560 the estimate recovers and alerts clear.

Be straight about the trade-off if asked: the confirmed alert came 28 h after the true
crossing versus 3 h for the conventional alarm — but the watch tier flagged it 1 h after,
and no confirmed alert was false.

### 3. Energy console (1–2 min)

Open `http://localhost:8501/energy_console`, press **Play** for a few ticks.

- "Same idea on Finnish residual load, 3 hours ahead: forecast, calibrated band, and a
  reserve recommendation expressed in fast-start units."
- Scroll to "Does the stated 90 % hold as conditions drift?" — the red line (un-calibrated)
  sinks to ~60 % in spring 2020; the green line (adaptive conformal) stays at 90 %.
  "This is the point: the guarantee survives drift without retraining."
- Do **not** claim reserve savings. Measured result: the dynamic rule keeps the risk level and
  cuts the 99th-percentile shortfall (141 vs 188 MW) but holds *more* reserve on average
  (396 vs 299 MW) than a fixed rule tuned in hindsight. Say: "with public data only, the
  intervals are honest but not yet sharp — sharper inputs are what a pilot would add."

### 4. The ask — Pilot proposal page (2 min)

- One subsystem, one historical dataset, non-production, under the UTU–Wärtsilä NDA template.
- Phase 1 data mapping → Phase 2 prototyping on CSC/LUMI → Phase 3 validation with operators.
- Then stop talking and go to the open questions.

## Discussion — questions to ask them

1. Which subsystem suffers most from alert fatigue or low trust in predictions today?
2. What false-alarm rate would make an operator actually act on an AI alert?
3. Where do remote-operations / expertise-centre engineers lose the most time per incident?
4. Is historical telemetry with maintenance outcomes available for a non-production pilot?
5. Who would be the right technical contact, and does this fit an existing Veturi work package?

## Likely questions and honest answers

- **"Is this real data?"** Marine: public simulator of a frigate gas-turbine plant (UCI,
  Coraddu et al. 2014) with sensor noise I added at typical instrument accuracy. Energy: real
  Finnish load and wind 2015–2020 (Open Power System Data). Manuals and logs: synthetic.
- **"Our engines are not gas turbines."** Correct — the method is model-agnostic: it wraps any
  predictor and needs only a held-out calibration set from the same asset family.
- **"What does the 90 % guarantee mean exactly?"** On average over operating points like the
  calibration data, at least ~90 % of intervals contain the true value. It is an average
  guarantee, not per-case, and it weakens if conditions drift — which is why the energy track
  uses the adaptive variant and why coverage is monitored (Trust page).
- **"Why did the marine interval deliver 88.7 %, not 90 %?"** Finite-sample variation with
  ~1 700 calibration and ~2 300 test points; the constant-width variant gives 90.1 % overall
  but drops to ~80 % at 3 kn, where the adaptive one holds 94 %.
- **"Is there an LLM?"** The demo runs offline with a deterministic extractive composer so
  nothing is hallucinated; an LLM endpoint can be plugged in, but the confidence statement is
  always computed, never generated.
- **"What is new here?"** Conformal prediction, RAG and ACI are published methods. The research
  contribution would be combining them into an operator-facing agent and validating the effect
  on decisions with real operational data.

## Confidentiality reminder (from the organiser's IPR note)

This is a non-confidential meeting. The demo uses only published methods and public data, so it
is safe to show. Do not describe unpublished details of your own research (e.g. your
metacognitive-architecture work) beyond a general introduction; propose an NDA for a deeper
follow-up.

## Next steps to propose

- Send the one-page summary and a link/recording of the demo within a week.
- A 45-minute technical follow-up under NDA with the person who owns the candidate dataset.
- Agree who is the contact on each side and a date.
