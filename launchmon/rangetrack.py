"""Track the ball's range, fit its flight and apply the acceptance tests.

    row, track = fit_range(prep, dmap, window, ball, p0)    # row["status"]: ACCEPT or REJECT

H1 not the first-after-boot mode, H2 no dropped frame in the shot, T1 enough points and a small
residual, T2 enough range span and a speed uncertainty of at most 2 %.
"""

import numpy as np

from .config import PARAMS
from .detect import line_search
from .flight import flight, geo_fit, cubic_speed


def pick(snr, f, rp, halfw, rbin, thr, nearest):
    """Ball range and SNR in frame f, from the peak within halfw of the predicted range rp,
    or None."""
    k_lo, k_hi = int((rp - halfw) / rbin), int((rp + halfw) / rbin) + 1
    gate = snr[f, k_lo:k_hi]
    if gate.max() < thr:
        return None
    if nearest:  # first local max within 6 dB of the gate max: the direct path, not a floor bounce
        candidates = [
            i
            for i in range(1, len(gate) - 1)
            if gate[i] >= gate[i - 1]
            and gate[i] >= gate[i + 1]
            and gate[i] >= gate.max() - 6
            and gate[i] >= thr
        ]
        i = candidates[0] if candidates else int(np.argmax(gate))
    else:
        i = int(np.argmax(gate))
    k = k_lo + i
    left, peak, right = snr[f, k - 1], snr[f, k], snr[f, k + 1]
    curvature = left - 2 * peak + right
    offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    return (k + np.clip(offset, -0.5, 0.5)) * rbin, peak


def track_t(snr, t, rbin, f0, v, fa, fb, nearest, rlim):
    """Track the ball over frames fa..fb with a gate that narrows over three passes around a
    robust quadratic prediction. Returns (frames, ranges, SNRs) or None."""
    coef = np.array([0.0, v, 2.2 - v * t[f0]])
    out = None
    for halfw, thr in PARAMS["gate_passes"]:
        frames, ranges, snrs = [], [], []
        for f in range(fa, fb):
            if not np.isfinite(snr[f]).all():
                continue
            predicted = np.polyval(coef, t[f])
            if predicted < rlim[0] or predicted > rlim[1]:
                continue
            found = pick(snr, f, predicted, halfw, rbin, thr, nearest)
            if found is None:
                continue
            frames.append(f)
            ranges.append(found[0])
            snrs.append(found[1])
        frames, ranges, snrs = map(np.array, (frames, ranges, snrs))
        if len(frames) < 5:
            continue
        keep = np.ones(len(frames), bool)
        for _ in range(3):
            fit = np.polyfit(t[frames[keep]], ranges[keep], 2)
            error = ranges - np.polyval(fit, t[frames])
            mad = np.median(np.abs(error[keep])) * 1.4826 + 1e-3
            keep = np.abs(error) < 3 * mad
        coef = fit
        out = (frames[keep], ranges[keep], snrs[keep])
    return out


def robust_geo(tf, rr, k, p0, it_max=8):
    """Drag + gravity fit, dropping points more than 3 MAD from it (10 mm floor) until the set
    of kept points stops changing."""
    keep = np.ones(tf.size, bool)
    seen = []
    for _ in range(it_max):
        (v0, el, t_launch), sig, rms = geo_fit(tf[keep], rr[keep], k, p0)
        error = rr - np.linalg.norm(flight(v0, el, 0, k, tf - t_launch, p0), axis=1)
        new = np.abs(error) < max(3 * 1.4826 * np.median(np.abs(error[keep])), 0.010)
        if (new == keep).all() or any((new == old).all() for old in seen) or new.sum() < 6:
            break
        seen.append(keep.copy())
        keep = new
    return keep, (v0, el, t_launch), sig, rms


def fit_range(prep, dmap, window, ball, p0):
    """Track, fit and test the ball inside window = (t0, t1). Returns (row, track), where
    track is dict(fr, rr, ss, fit) or None."""
    snr, t, valid, rbin = dmap, prep.t, prep.valid, prep.rbin
    k = ball.k_drag
    band = ball.band_m
    row = dict(window=list(window))
    clipped = np.clip(np.nan_to_num(snr, nan=0.0), 0, 25)
    first, last = np.searchsorted(t, window[0]), np.searchsorted(t, window[1])
    score, f0, v = line_search(clipped[first:last], t[first:last], rbin, vmin=ball.v_min_seed)
    f0 += first
    row.update(line_score=round(score, 2), v_line=v, t_seed=round(float(t[f0]), 3))
    found = track_t(
        snr, t, rbin, f0, v, first, last, ball.nearest_peak, rlim=(band[0] - 0.2, band[1] + 0.1)
    )
    reasons = ["H1 282 Hz first-after-boot mode"] if prep.first_after_boot else []
    if found is None:
        row.update(status="REJECT", reasons=reasons + ["no track"])
        return row, None
    fr, rr, ss = found
    in_band = (rr >= band[0]) & (rr <= band[1])
    fr, rr, ss = fr[in_band], rr[in_band], ss[in_band]
    if fr.size:
        all_valid = valid[fr.min() : fr.max() + 1].all()
        contiguous = (
            np.diff(np.round(t[fr.min() : fr.max() + 1] / np.median(np.diff(t)))) == 1
        ).all()
    else:
        all_valid = contiguous = False
    if not (all_valid and contiguous):
        reasons.append("H2 invalid/dropped frame inside the shot")
    n_min = PARAMS["min_points"]
    if fr.size < n_min:
        row.update(
            status="REJECT", reasons=reasons + ["T1 n=%d < %d" % (fr.size, n_min)], n=int(fr.size)
        )
        return row, dict(fr=fr, rr=rr, ss=ss, fit=None)
    tf = t[fr]
    keep, (v0, el, t_launch), sig, rms = robust_geo(tf, rr, k, p0)
    n_before = fr.size
    fr, rr, ss, tf = fr[keep], rr[keep], ss[keep], tf[keep]
    row["n_dropped_fit"] = int(n_before - keep.sum())
    if fr.size < n_min:
        row.update(
            status="REJECT",
            reasons=reasons + ["T1 n=%d < %d after outlier rejection" % (fr.size, n_min)],
            n=int(fr.size),
        )
        return row, dict(fr=fr, rr=rr, ss=ss, fit=None)
    v_cubic = cubic_speed(tf, rr)
    floor = PARAMS["sig_r_floor"]
    # rescale to a residual of at least the floor
    sigma_v = sig[0] * max(rms, floor) / max(rms, 1e-9)
    span = rr.max() - rr.min()
    if rms > ball.rms_limit_m:
        reasons.append("T1 rms %.1f mm > %.0f" % (rms * 1e3, ball.rms_limit_m * 1e3))
    if span < PARAMS["min_span_m"]:
        reasons.append("T2 span %.2f m < %.1f" % (span, PARAMS["min_span_m"]))
    if not np.isfinite(sigma_v) or sigma_v / v0 > PARAMS["max_sv_frac"]:
        reasons.append(
            "T2 sigma_v %.1f %% > %.0f" % (100 * sigma_v / v0, 100 * PARAMS["max_sv_frac"])
        )
    if not (np.isfinite(v0) and v0 > 0):
        reasons.append("fit failed")
    row.update(
        status="ACCEPT" if not reasons else "REJECT",
        reasons=reasons,
        n=int(fr.size),
        r0=round(float(rr.min()), 2),
        r1=round(float(rr.max()), 2),
        span=round(float(span), 2),
        snr_med=round(float(np.median(ss)), 1),
        v_cubic=round(float(v_cubic), 2),
        v_geo=round(float(v0), 2),
        sv_pct=round(float(100 * sigma_v / v0), 2),
        el_range_only=round(float(np.degrees(el)), 1),
        rms_mm=round(float(rms * 1e3), 1),
        t_launch=round(float(t_launch), 4),
    )
    return row, dict(fr=fr, rr=rr, ss=ss, fit=(v0, el, t_launch))
