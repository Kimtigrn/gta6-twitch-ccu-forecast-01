# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kimtigrn/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kimtigrn/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day, 2026-11-19 (UTC), after the first hour, as five quantiles
per hour. Entry for the MakeFun bounty
`gta6-twitch-ccu-forecast-01` (pinball loss on the log scale; a prize needs a
loss strictly below the best house entry).

## Run it

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.05 s per case on one CPU, deterministic (two runs byte-identical).
No dependencies beyond the Python 3.13 standard library.

## What it does

`main.py` (Python 3.13, standard library only, no network) reads one case file
`/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes `/output/<id>.out` (or
`$MAKEFUN_OUTPUT_DIR`): 23 rows, hours 2–24, each five quantiles
`[q05, q25, q50, q75, q95]` of that hour's Twitch viewers.

The model, in one paragraph: the launch-day curve is modelled in logs as ratios
to the hour-1 print, from the comparable past launches shipped in the case file.
Per hour, two small log-space OLS regressions are fit on clean big launches
(hour-1 ≥ 10,000 viewers, no early access, no weak tier) — one on
`log(hour-1)` and the pre-release evening mean, one on `log(hour-1)` alone —
and blended adaptively: the pre-release signal is trusted only while
pre/hour-1 stays inside the range seen among big clean launches
([0.4, 4]); outside it the forecast is the geometric mean of the two models
with widened bands. A correction for the US-East midnight unlock (+0.38 log at
that hour, +0.33 the next, the hour computed from the release date by the US
daylight-saving rule) fixes the regression's systematic under-prediction of the
unlock rise. Bands are normal quantiles in log space with sigma = fitted
residual SD × 0.70. Full reasoning, variants tried, and risks: see `NOTES.md`.

## Practice score (leave-one-out, honest)

Scored locally against the four practice launches' actual release-day curves
(Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's
Spider-Man 2). Each case was forecast using only its own case file's
comparables — never the other practice launches' curves. The local scorer
implements the pinball loss exactly as `RULES.txt` specifies and reproduces
the house baseline's published practice numbers (0.387 / 82.8% / 0.783).

| entry | loss | typical miss | 90% coverage |
|---|---|---|---|
| house baseline (starter) | 0.387097 | 82.75% | 0.783 |
| **this entry** | **0.141448** | **23.47%** | **0.957** |

Per case: p00 0.157603, p01 0.110597, p02 0.128150, p03 0.169441.

With the final's training set (the 4 practice launches added as comparables,
41 in all, simulated from their public curves) the same program scores
0.138669 — the final runs with 42 comparables, so this is the closer
analogue. A practice score is no guide to the final: the final is one fresh
case (GTA VI itself).

## Files

- `main.py` — the entry (single file, no other modules).
- `LICENSE` — MIT (as shipped in the kit's starter).
- `README.md` — this file.
- `NOTES.md` — method, variants tried with their practice losses, why the
  chosen model should generalise, known risks for the final.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
