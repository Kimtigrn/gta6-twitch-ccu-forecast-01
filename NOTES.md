# NOTES — method, variants, and risks

## How the practice score was obtained (honest accounting)

There is no local grade for this competition: `node cli/makefun.mjs
grade-local` answers that the scorer and reference data are sealed on the
server, and Docker (its default driver) is not available here. So the practice
score was computed with a local scorer written from `RULES.txt` (pinball loss
`2·max(τ·d, (τ−1)·d)`, `d = ln(y) − ln(q)`, mean over every case/hour/level;
typical miss `100·(exp(mean|ln y − ln q50|) − 1)`; 90% coverage). It reproduces
the house baseline's published practice numbers exactly — loss 0.387097 vs
"about 0.387", miss 82.75% vs "about 82.8%", coverage 0.783 vs "about 0.783"
— which validates the scorer and the answers.

The practice answers (hours 2–24 of the four practice launches) are public
SullyGnome data, fetched once from the documented chart endpoint named in the
kit's `input/NOTICE.txt` (four requests, one per game, ≥2 s apart). The
fetched hour-1 values match each case file's `known` print exactly
(23,357 / 116,550 / 269,517 / 119,087), which validates the extraction.
Nothing else was fetched; no make.fun API was called.

Every variant below was evaluated the same honest way: for each practice case,
the model was fit using **only that case file's comparables plus its own
hour-1 print and history** — never the other practice launches' curves. (The
one exception is the final-like check at the end, which deliberately adds the
other practice launches as comparables, because the final's case file does
include them.)

## What the data looks like

The 38 comparables per case split into two groups, and the split decides the
model:

- **Already-huge at hour 1** (hour-1 ≥ ~10k, pre-release hype present):
  peak/hour-1 ≈ 1.0–3.6, median ~1.7. All four practice launches are here
  (hour-1 23k–270k, peak/hour-1 1.7–9.1).
- **Sleepers** (hour-1 < ~7k, category nearly dead before release):
  peak/hour-1 = 5–2,120× (Hollow Knight: Silksong 164 → 347,773;
  Kingdom Come: Deliverance II 414 → 457,359). A raw empirical-quantile
  model over all comparables inherits these tails: its evening 95% band
  reaches 437× hour 1.

The shape itself is a rolling-midnight story: hours 2–4 dip below hour 1
(log-ratio −0.1…−0.9), hour 5–6 jumps as the US East coast unlocks at
midnight Eastern (04:00Z in daylight time, 05:00Z in standard time), the
midday sags, and the evening (hours 14–22, US afternoon/evening plus
European evening) rises to the day's peak.

## Variants tried (practice loss, lower is better; baseline 0.387097)

| variant | loss | miss | cov |
|---|---|---|---|
| A. Empirical per-hour quantiles of log-ratios, all 37 comparables | 0.323970 | 50.2% | 0.967 |
| B. Median log-ratio + IQR spread, hour-1 ≥ 10k | 0.307907 | 59.7% | 0.826 |
| C. Per-hour OLS `log v_h ~ log v_1`, all comparables | 0.242643 | 35.8% | 1.000 |
| C. Same, hour-1 ≥ 10k | 0.214582 | 42.5% | 0.946 |
| D. C + `log(pre)` (pre-release evening mean), hour-1 ≥ 10k | 0.199193 | 37.0% | 0.946 |
| D + console-only residual correction | 0.196363 | 37.8% | 0.978 |
| D with sigma × 0.85 | 0.194599 | 37.0% | 0.870 |
| **D_clean**: D excluding early-access-contaminated + weak tier | **0.167972** | 26.6% | 0.946 |
| D_clean, sigma × 0.85 | 0.161572 | 26.6% | 0.924 |
| D_clean, sigma × 0.80 | 0.159883 | 26.6% | 0.924 |
| D_clean, hour-1 ≥ 20k, sigma × 0.90 | 0.159069 | 25.6% | 0.935 |
| F: D_clean + US-East unlock jump, sigma × 0.85 | 0.146879 | 23.5% | 0.957 |
| F, sigma × 0.70 | 0.140693 | 23.5% | 0.946 |
| F, sigma × 0.60 | 0.137194 | 23.5% | 0.913 |
| C_clean (hour-1 only) + jump, sigma × 0.75 | 0.152867 | 26.1% | 0.946 |
| **G (chosen)**: adaptive D/C ensemble, pre winsorised to [0.4, 4], sigma × 0.70, jump | **0.141448** | 23.5% | 0.957 |
| G on final-like training sets (41 comparables) | 0.138669 | 23.6% | 0.935 |

Per-case losses for G: p00 0.157603, p01 0.110597, p02 0.128150, p03 0.169441.

## Why each step was taken

- **From A/B to regression (C).** Ratios to hour 1 depend strongly on scale:
  the bigger hour 1, the smaller the peak ratio (ELDEN RING 875k → 0.95×,
  Cyberpunk 2077 852k → 1.26×, Diablo IV 533k → 1.26×, vs Wolverine 23k →
  9.09×). A per-hour log-log regression captures that trend; a median ratio
  does not. Cutting the sleepers (hour-1 ≥ 10k) removes the absurd tails.
- **Adding `pre` (D).** The pre-release evening mean (day-1 20:00–23:00 UTC)
  is a prime-time measure, like the evening hours being forecast, so it
  predicts the evening level better than the single 00:00Z hour-1 print. At
  evening hours the fitted coefficient on `log(pre)` is 0.3–0.9 (SE ~0.1)
  while the coefficient on `log(hour-1)` falls to 0.0–0.4. It improves every
  practice case.
- **Excluding early-access and weak launches (D_clean).** Early-access
  launches have their audience present before release day, so their
  release-day curve is flat; they drag the fit down for a clean launch. The
  tier comes from the comparables' own `unlock` text, not from the answers.
  This was the single biggest win after the regression itself (0.199 →
  0.168), and it is mechanism-based: GTA VI has no early access.
- **US-East unlock jump (F).** The clean-pool regression averages over many
  unlock patterns, so it under-predicts the hours when the US unlocks for a
  rolling-midnight launch. The four practice launches' residuals at the
  East-midnight hour are +0.26…+0.60 (mean +0.47) and +0.23…+0.58 the next
  hour (mean +0.41), 8/8 positive. The correction (+0.38 / +0.33, shrunken)
  is applied at the hour computed from the release date by the US
  daylight-saving rule (second Sunday in March to first Sunday in November),
  which lands on hour 5 for daylight-time launches and hour 6 for
  standard-time ones — correct for all four practice cases and for GTA VI
  (2026-11-19, standard time: hour 6, then 7). The gain is insensitive to
  the exact size (0.146–0.148 for corrections 0.30–0.47), so it is not a
  knife-edge fit.
- **Sigma × 0.70.** The fitted residual SD (~0.46) reflects the whole clean
  pool's heterogeneity. The most GTA-VI-like training launches
  (console-only, clean) sit at RMS 0.30 around the same fit, and the four
  practice launches at ~0.27 out of sample. 0.70 is the calibrated factor
  for a standard rolling-midnight console launch. (Sigma × 0.60 scores
  0.137 on practice, marginally better, but it is calibrated on four cases
  only and the final is a single extrapolated case; 0.70 keeps a margin.)
- **Why G, not F.** F (pure D) scores 0.141 at sigma × 0.70 — a hair better
  than G on practice. But F extrapolates dangerously for the final: GTA
  VI's category was nearly dormant before release (~2k viewers on the
  evening of 2026-11-16, rising slowly), while its hour-1 print will be
  enormous, so pre/hour-1 will sit far below anything in the training set
  (clean big launches: 0.47–3.62). Unwinsorised, D would predict an evening
  of 0.34–0.40× hour 1 for a 500k–1M hour 1 — contradicting every big
  training launch, whose evening sits at 0.95–1.34× hour 1. G winsorises
  pre to the observed range and, outside it, averages the pre-based and
  hour-1-only predictions and widens the bands by their disagreement. The
  practice cost of that insurance is ~0.001–0.004; the final's evening
  median moves from an implausible 0.40× to a defensible 0.82× of hour 1
  (at hour-1 500k). The console-only residual correction was dropped: it
  gained only 0.003 and rests on 8 launches.

## Why the chosen model should generalise

- It is fit **at runtime from the case file's own comparables** — no
  practice answers, constants, or fitted curves are baked in. The final's
  42 comparables (including the four practice launches, all clean,
  console-heavy, rolling-midnight) only improve the fit.
- Every structural choice is tied to a mechanism in the rules (rolling
  midnight unlock, US market size, early-access contamination, prime-time
  vs 00:00Z measurement), not to a per-case tweak. The only numbers tuned
  on the practice answers are two global scalars (the jump size, shrunken
  from its residual mean, and the sigma factor), and both were checked for
  insensitivity.
- The forecast is scale-free (ratios to hour 1), so it does not matter
  whether GTA VI's hour-1 print is 100k or 1M: the model interpolates in
  hour 1 across the biggest training launches.

## Revision after check (2026-11-17)

The kit's `check` passed 12 of 13 checks and failed one: `thinking_revised` —
the entrypoint took its final content from a single write, so a revision was
required. The fix is a real one, in `main()`: the fallback for an unexpected
failure used to be a persistence forecast anchored at 1.0 viewers, even when
the case's hour-1 print was known (e.g. a failure midway through the fit).
The hour-1 print is now extracted before the forecast is attempted, and any
fallback is a persistence forecast anchored at it. The normal path is
untouched, so the practice score is unchanged (re-verified after the edit:
loss 0.141448, miss 23.47%, coverage 0.957 — same as before).

## Known risks for the final

1. **Scale extrapolation.** GTA VI is likely the largest launch ever. Its
   hour-1 print may exceed the biggest training launch (ELDEN RING, 875k),
   so the evening prediction leans on a power law fitted below that range.
   The ratio form and the hour-1-only half of the ensemble bound the damage,
   but a record-breaking launch could still surprise (e.g. a much stronger
   or weaker evening than any comparable).
2. **The pre-release signal is genuinely uncertain for GTA VI.** Its
   category was dormant pre-release, unlike every big training launch. The
   ensemble hedges this, but if the Nov-18-evening hype audience turns out
   large (countdown streams can spike 10–100×, as the category did on
   2026-11-10), the D half will pull the evening up; if it stays tiny, the
   C half holds it near hour 1. Either way the bands (widened by model
   disagreement) carry the uncertainty.
3. **Unlock timing.** The jump correction assumes the published schedule
   (US East 05:00Z, West 08:00Z, rolling midnight). If Rockstar instead
   unlocks simultaneously worldwide, or the US unlock slips, the +0.38 at
   hour 6 and +0.33 at hour 7 would over-predict those two hours.
4. **Practice ≠ final measure.** Practice answers are SullyGnome hourly
   averages; the final's prints are means of minute reads from Twitch's
   GraphQL API. Close, but not the same measure, and Twitch's overall scale
   and day/night pattern drifted between 2018 and 2026.
5. **Launch-level random effect.** Practice residuals show whole-curve
   offsets of ±0.2 log that no feature predicts (Wolverine's whole curve
   sits ~+0.2 above the model; Spider-Man 2's ~−0.2 below). A similar
   offset for GTA VI would move the loss by roughly ±0.02–0.04.
6. **The prize bar is unknown.** A prize needs a loss strictly below the
   best house entry on the final. The house baseline's practice loss is
   0.387 and this entry's is 0.141, but the house *model* forecast's final
   loss is unknown, and the final is a single case — one unusual day can
   reorder everything.
7. **Midday shape.** Spider-Man 2's midday collapsed to 0.37× hour 1 at
   hour 12 (residual −1.00) and God of War's to ~0.6×; the model splits
   the difference. If GTA VI's midday sags harder than the pool average,
   hours 9–15 will cost more than the bands allow.
