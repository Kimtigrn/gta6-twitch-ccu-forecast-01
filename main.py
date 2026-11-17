"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out
(else $MAKEFUN_OUTPUT_DIR): for each of hours 2..24 of the release day (UTC), five
quantiles [q05, q25, q50, q75, q95] of that hour's Twitch viewers in the game's
category. Python 3.13, standard library only, no network, deterministic.

Method (see NOTES.md for the full reasoning and the variants that lost)
------------------------------------------------------------------------
The launch-day curve is modelled in logs, as ratios to the hour-1 print, from the
comparable past launches shipped in the case file. Per hour h = 2..24 two small
ordinary-least-squares regressions are fit in log space on the same training set:

    D:  log viewers_h ~ a + b * log(hour-1 viewers) + c * log(pre)
    C:  log viewers_h ~ a + b * log(hour-1 viewers)

where `pre` is the case's own pre-release evening mean (the mean viewers of the
four hours before midnight on the day before release, 20:00-23:00 UTC) and the
comparables' `pre` is the same window of their curves. `pre` is a prime-time
measure, like the evening hours being forecast, so it carries real signal about a
launch's scale beyond the single 00:00Z hour-1 print (on the practice launches
the evening level is better predicted by `pre` than by hour 1 alone).

Training set: comparables with hour-1 viewers >= 10,000, a pre-release evening
mean, and a clean tier. Two groups are excluded, on mechanism, not on outcome:
  * early-access-contaminated launches (their `unlock` text says so): their
    audience was already present before release day, so their release-day curve
    is flat and drags every hour's fit down for a clean launch like GTA VI.
  * "weak" tier launches (free-to-play launches whose shape is not a paid
    launch's) and sleeper launches with a tiny hour-1 print (their category was
    essentially dead at hour 1, so their ratios to hour 1 run into the hundreds
    and would wreck a log-ratio fit).
The Grand Theft Auto VI category's own pre-release history, shipped as a
comparable, has no hour 1 and is excluded by the hour-1 rule automatically.

The pre-release signal is trusted only inside the range seen among big clean
launches (pre / hour-1 in [0.4, 4]). `pre` is winsorised to that band, and when
the raw ratio falls outside it the median becomes the geometric mean of the D
and C predictions (weight 0.5 each) and the bands are widened by
0.25 * |log D - log C|, the two models' disagreement. This matters for the
final: GTA VI's category was nearly dormant before release (a few thousand
viewers) while its hour-1 print will be enormous, so its pre/hour-1 ratio will
sit far below anything in the training set, and an unwinsorised pre-based
forecast would collapse the evening. The hour-1-only model, by contrast,
interpolates in hour 1, where the biggest training launches (ELDEN RING,
Cyberpunk 2077, Diablo IV) sit.

US-East unlock jump. The game unlocks at midnight local, rolling, and the US
East coast is the largest Twitch market: it unlocks at midnight Eastern, which
is 04:00Z while US daylight saving is in effect and 05:00Z after it ends (first
Sunday in November). Hour 1 (00:00-01:00Z) therefore has no US audience yet.
The clean-pool regression averages over launches with many unlock patterns, so
it systematically under-predicts the unlock hours for a rolling-midnight
launch: on the four practice launches the residual at the East-midnight hour is
+0.26..+0.60 (mean +0.47) and at the next hour +0.23..+0.58 (mean +0.41), 8/8
positive. A shrunken correction of +0.38 log at that hour and +0.33 the next
is added; the hour itself is computed from the case's release date by the US
daylight-saving rule, so it lands on hour 5 for daylight-time launches and
hour 6 for standard-time ones (GTA VI, 2026-11-19, is standard time: hour 6,
then hour 7).

Spread. The bands are normal quantiles around the median in log space with
sigma = max(residual SD of D, residual SD of C) * 0.70, floored at 0.20. The
fitted residual SD (~0.46) reflects the whole clean pool's heterogeneity; the
most GTA-VI-like training launches (console-only, clean) sit at RMS 0.30
around the same fit, and the four practice launches at ~0.27 out of sample, so
0.70 is the calibrated factor for a standard rolling-midnight console launch.

Fallbacks. If the case has no usable pre-release mean or too few clean
comparables, the hour-1-only regression is used; if that too is unavailable, a
persistence forecast with a widening band is written. Any unexpected failure
for a case still writes a valid persistence forecast, because an invalid case
scores as the hour-1 print and takes the entry out of the prizes, while a
valid-but-mediocre forecast does not.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

H1_MIN = 10000.0          # hour-1 viewers a comparable needs to anchor a ratio fit
CLIP_LO, CLIP_HI = 0.4, 4.0   # trusted range of pre / hour-1 among big clean launches
SIGMA_MULT = 0.70         # calibration of the fitted residual SD (see docstring)
SIGMA_FLOOR = 0.20
SIGMA_CAP = 1.20
JUMP1, JUMP2 = 0.38, 0.33     # US-East unlock correction, log units
MIN_ROWS = 8              # rows a per-hour regression needs to be trusted


def parse_time(text):
    """ISO 8601 with a trailing Z into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def median(vals):
    return quantile(sorted(vals), 0.5)


def solve_linear(mat, vec):
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular system")
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))) / aug[i][i]
    return out


def ols(rows_x, rows_y):
    """Least squares for rows_x (each a list, intercept included by the caller).

    Returns (beta, residual_sd)."""
    k = len(rows_x[0])
    n = len(rows_x)
    mat = [[sum(rows_x[r][i] * rows_x[r][j] for r in range(n)) for j in range(k)]
           for i in range(k)]
    vec = [sum(rows_x[r][i] * rows_y[r] for r in range(n)) for i in range(k)]
    for i in range(k):
        mat[i][i] += 1e-9
    beta = solve_linear(mat, vec)
    resid = [rows_y[r] - sum(beta[j] * rows_x[r][j] for j in range(k)) for r in range(n)]
    dof = max(n - k, 1)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return beta, sd


def is_clean(comp):
    """True unless the launch is early-access-contaminated or weak tier.

    The tier is read from the comparable's own `unlock` text, which carries a
    "Tier: ..." note. Early-access launches have their audience present before
    release day, so their flat release-day shape does not describe a clean
    launch; weak-tier launches (free-to-play) are not paid-launch analogues."""
    unlock = comp.get("unlock") or ""
    if "early-access-contaminated" in unlock:
        return False
    if "Tier: weak" in unlock:
        return False
    return True


def extract_comparables(case):
    """Release-day curves and pre-release evening means of the usable comparables.

    Returns a list of dicts {hours: {1..24: viewers}, h1, pre} where `pre` is
    the mean viewers of the four hours before midnight on the day before
    release (20:00-23:00 UTC). A comparable without an hour-1 print (such as
    the GTA VI category's own pre-release history) is skipped."""
    out = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = {}
        pre_vals = []
        for row in comp.get("hourly") or []:
            v = row.get("viewers")
            if not v or v <= 0:
                continue
            try:
                t = parse_time(row["from"])
            except (KeyError, TypeError, ValueError):
                continue
            d = (t - day_start).total_seconds() / 3600.0
            if 0 <= d < 24:
                hours[int(d) + 1] = float(v)
            elif -4 <= d < 0:
                pre_vals.append(float(v))
        if 1 not in hours:
            continue
        out.append({
            "hours": hours,
            "h1": hours[1],
            "pre": (sum(pre_vals) / len(pre_vals)) if pre_vals else None,
            "clean": is_clean(comp),
        })
    return out


def case_pre_release(case):
    """The case's own pre-release evening mean (day-1 20:00-23:00 UTC).

    From history_hourly when present, else from the raw history points."""
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    vals = []
    for row in case.get("history_hourly") or []:
        v = row.get("viewers")
        if not v or v <= 0:
            continue
        try:
            t = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if -4 <= (t - day_start).total_seconds() / 3600.0 < 0:
            vals.append(float(v))
    if vals:
        return sum(vals) / len(vals)
    for row in case.get("history") or []:
        v = row.get("viewers")
        if not v or v <= 0:
            continue
        try:
            t = parse_time(row["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if -4 <= (t - day_start).total_seconds() / 3600.0 < 0:
            vals.append(float(v))
    return (sum(vals) / len(vals)) if vals else None


def case_hour1(case):
    """Hour-1 print, at least 1. Falls back to the last history point, then 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    hist = case.get("history_hourly") or case.get("history") or []
    for row in reversed(hist):
        if row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    return 1.0


def us_east_unlock_hour(day_start):
    """The UTC hour (1-based) in which midnight Eastern falls on the release date.

    US daylight saving runs from the second Sunday in March to the first Sunday
    in November: midnight Eastern is 04:00Z in daylight time (hour 5) and
    05:00Z in standard time (hour 6)."""
    try:
        ds = parse_time(day_start).date()
    except (KeyError, TypeError, ValueError):
        return 6

    def nth_sunday(year, month, n):
        first = datetime(year, month, 1).date()
        first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
        return first_sunday + timedelta(weeks=n - 1)

    dst_start = nth_sunday(ds.year, 3, 2)
    dst_end = nth_sunday(ds.year, 11, 1)
    return 5 if dst_start <= ds < dst_end else 6


def persistence_forecast(p1):
    """The house baseline shape: every hour at the hour-1 print, spread widening
    like a random walk in log space. Used only as a fallback."""
    return {
        "hours": [
            {"hour": h,
             "q": [round2(max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))) for z in Z]}
            for h in range(2, 25)
        ]
    }


def fit_hour(comps, hour, use_pre):
    """One hour's OLS fit. Returns (beta, sigma) or None when too few rows."""
    if use_pre:
        rows = [(math.log(c["h1"]), math.log(c["pre"]), math.log(c["hours"][hour]))
                for c in comps if hour in c["hours"] and c["pre"]]
        if len(rows) < MIN_ROWS:
            return None
        x = [[1.0, a, b] for a, b, _ in rows]
        y = [c for _, _, c in rows]
    else:
        rows = [(math.log(c["h1"]), math.log(c["hours"][hour]))
                for c in comps if hour in c["hours"]]
        if len(rows) < MIN_ROWS:
            return None
        x = [[1.0, a] for a, _ in rows]
        y = [b for _, b in rows]
    try:
        beta, sd = ols(x, y)
    except ValueError:
        return None
    return beta, sd


def forecast_case(case):
    p1 = case_hour1(case)
    comps = extract_comparables(case)
    clean = [c for c in comps if c["clean"] and c["h1"] >= H1_MIN and c["pre"]]
    big = [c for c in comps if c["h1"] >= H1_MIN]
    pre = case_pre_release(case)

    med_log_ratio = {}
    sigma = {}

    if len(clean) >= 10 and pre:
        # Full model: D (hour-1 + pre) blended with C (hour-1 only).
        raw_ratio = pre / p1
        use_full_d = CLIP_LO <= raw_ratio <= CLIP_HI
        weight_d = 1.0 if use_full_d else 0.5
        log_pre = math.log(pre)
        log_pre = min(max(log_pre, math.log(p1) + math.log(CLIP_LO)),
                      math.log(p1) + math.log(CLIP_HI))
        log_p1 = math.log(p1)
        for h in range(2, 25):
            fit_d = fit_hour(clean, h, use_pre=True)
            fit_c = fit_hour(clean, h, use_pre=False)
            if fit_d is None and fit_c is None:
                med_log_ratio[h] = 0.0
                sigma[h] = 0.5
                continue
            if fit_d is None:
                beta_c, sd_c = fit_c
                med_log_ratio[h] = beta_c[0] + beta_c[1] * log_p1 - log_p1
                sigma[h] = max(SIGMA_FLOOR, sd_c * SIGMA_MULT)
                continue
            beta_d, sd_d = fit_d
            m_d = beta_d[0] + beta_d[1] * log_p1 + beta_d[2] * log_pre
            if fit_c is None:
                med_log_ratio[h] = m_d - log_p1
                sigma[h] = max(SIGMA_FLOOR, sd_d * SIGMA_MULT)
                continue
            beta_c, sd_c = fit_c
            m_c = beta_c[0] + beta_c[1] * log_p1
            med_log_ratio[h] = weight_d * m_d + (1.0 - weight_d) * m_c - log_p1
            sig = max(sd_d, sd_c) * SIGMA_MULT
            if weight_d < 1.0:
                # The two models disagree exactly where the pre signal is out
                # of support; widen the band by their disagreement.
                sig += 0.25 * abs(m_d - m_c)
            sigma[h] = min(SIGMA_CAP, max(SIGMA_FLOOR, sig))
    elif len(big) >= MIN_ROWS:
        # No usable pre-release mean: hour-1-only regression on the big launches.
        log_p1 = math.log(p1)
        for h in range(2, 25):
            fit_c = fit_hour(big, h, use_pre=False)
            if fit_c is None:
                med_log_ratio[h] = 0.0
                sigma[h] = 0.5
                continue
            beta_c, sd_c = fit_c
            med_log_ratio[h] = beta_c[0] + beta_c[1] * log_p1 - log_p1
            sigma[h] = min(SIGMA_CAP, max(SIGMA_FLOOR, sd_c * SIGMA_MULT))
    else:
        # Too little training data to fit anything: persistence fallback.
        return persistence_forecast(p1)

    # US-East unlock jump (see module docstring): the East coast unlocks at
    # midnight Eastern, hour 5 in daylight time and hour 6 in standard time.
    east_hour = us_east_unlock_hour(case.get("day_start", ""))
    if east_hour in med_log_ratio:
        med_log_ratio[east_hour] += JUMP1
    if east_hour + 1 in med_log_ratio:
        med_log_ratio[east_hour + 1] += JUMP2

    hours = []
    for h in range(2, 25):
        q = []
        prev = 0.0
        for z in Z:
            value = p1 * math.exp(med_log_ratio[h] + z * sigma[h])
            value = min(1e9, max(1.0, value))
            value = round2(value)
            # Rounding is monotone, but enforce non-decreasing defensively.
            if value < prev:
                value = prev
            prev = value
            q.append(value)
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without a valid forecast.
            forecast = persistence_forecast(1.0)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
