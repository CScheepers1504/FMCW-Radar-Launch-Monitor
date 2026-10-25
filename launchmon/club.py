"""Club-head speed from the swing's return, golf only.

    club = club_speed(prep, t_launch, ball_curve, radar_height_m)    # dict, or None if not found

The ball is masked out, the head's outer edge is picked before and after impact, and the steepest
slope of a Gompertz curve fitted to the edges, corrected for the viewing angle, is the speed. The
attack angle is assumed 0 +- 5 deg, and there is no reference to validate against.
"""

import numpy as np

from .config import PARAMS
from .flight import least_squares

CURVE_LO = np.array([0.0, 1.0, -0.05, 0.002])  # a, B, tc, tau: the range the curve may take
CURVE_HI = np.array([1.0, 5.0, 0.06, 0.15])
CURVE_START = np.array([0.35, 2.8, 0.0, 0.025])
BALL_MASK_BINS = 3  # bins either side of the ball's track removed


def gompertz(t, a, B, tc, tau):
    return a + B * np.exp(-np.exp(-(t - tc) / tau))


def edge_pick(col, lo, hi, thr, far):
    """Edge of the strongest return in col[lo:hi], walking away from the radar (far) or towards
    it while above thr. In bins, or None."""
    gate = col[lo:hi]
    if not gate.size:
        return None
    k = int(np.argmax(gate))
    if gate[k] < thr + 3:  # needs a real return in the gate
        return None
    step = 1 if far else -1
    while 0 <= k + step < gate.size and gate[k + step] >= thr:
        k += step
    if not 0 <= k + step < gate.size:
        return float(lo + k)
    return lo + k + step * (gate[k] - thr) / (gate[k] - gate[k + step])


def _fit(t, r, start):
    """Gompertz least-squares fit, with the parameters clipped to their allowed range."""

    def residuals(q):
        return gompertz(t, *np.clip(q, CURVE_LO, CURVE_HI)) - r

    x = least_squares(residuals, start, x_scale=[0.1, 0.5, 0.01, 0.01]).x
    return np.clip(x, CURVE_LO, CURVE_HI)


def _edges(t_rel, snr_map, rbin, thr):
    """Edge picks and the fitted curve, the gate narrowing pass by pass."""
    t_lo, t_hi = PARAMS["club_window_s"]
    curve, edges = CURVE_START.copy(), np.empty((0, 2))
    for gate in PARAMS["club_gates_m"]:
        predicted = gompertz(t_rel, *curve)
        edges = []
        for k in np.flatnonzero((t_rel >= t_lo) & (t_rel <= t_hi)):
            lo = max(int((predicted[k] - gate) / rbin), 1)
            hi = int((predicted[k] + gate) / rbin) + 1
            edge = edge_pick(snr_map[k], lo, hi, thr, far=t_rel[k] >= 0)
            if edge is not None:
                edges.append((t_rel[k], edge * rbin))
        edges = np.array(edges).reshape(-1, 2)
        if len(edges) < PARAMS["club_min_edges"]:
            return None, edges
        curve = _fit(edges[:, 0], edges[:, 1], curve)
    return curve, edges


def _steepest(curve):
    """Time and value of the curve's steepest slope inside the fitted window."""
    t_lo, t_hi = PARAMS["club_window_s"]
    tg = np.linspace(t_lo, t_hi, 2701)
    slope = np.gradient(gompertz(tg, *curve), tg)
    i = int(np.argmax(slope))
    return float(tg[i]), float(slope[i])


def club_speed(prep, t_launch, ball_curve, radar_height_m, seed=1):
    """t_launch: ball launch in capture time. ball_curve: dict(t, r) of the fitted ball range,
    masked out of the map after launch."""
    t, rbin = prep.t, prep.rbin
    map_lo, map_hi = PARAMS["club_map_s"]
    i0, i1 = np.searchsorted(t, t_launch + map_lo), np.searchsorted(t, t_launch + map_hi)
    t_rel = t[i0:i1] - t_launch
    snr_map = np.nan_to_num(prep.snr[i0:i1]).copy()
    ball_r = np.interp(
        t_rel + t_launch, ball_curve["t"], ball_curve["r"], left=np.nan, right=np.nan
    )
    for k in range(len(t_rel)):
        if t_rel[k] >= 0 and np.isfinite(ball_r[k]):
            j = int(round(ball_r[k] / rbin))
            snr_map[k, max(j - BALL_MASK_BINS, 0) : j + BALL_MASK_BINS + 1] = -50.0
    snr_map = snr_map - np.median(snr_map, 0)[None]

    curve, edges = _edges(t_rel, snr_map, rbin, PARAMS["club_thr_db"])
    if curve is None:
        return None
    t_peak, radial = _steepest(curve)
    curve_alt, _ = _edges(t_rel, snr_map, rbin, PARAMS["club_thr_alt_db"])
    radial_alt = _steepest(curve_alt)[1] if curve_alt is not None else np.nan

    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(PARAMS["club_boot"]):
        i = rng.integers(0, len(edges), len(edges))
        resampled.append(_steepest(_fit(edges[i, 0], edges[i, 1], curve))[1])
    edge_sd = float(np.std(resampled))

    r_peak = float(gompertz(t_peak, *curve))
    depression = float(
        np.degrees(
            np.arcsin(np.clip((radar_height_m - PARAMS["club_head_height_m"]) / r_peak, -1, 1))
        )
    )
    alpha = PARAMS["club_alpha_deg"]

    def true_speed(attack_deg):
        return radial / np.cos(np.radians(depression + attack_deg))

    speed, speed_lo, speed_hi = true_speed(0.0), true_speed(-alpha), true_speed(alpha)
    half_thr = 0.5 * abs(radial_alt / radial - 1) if np.isfinite(radial_alt) else 0.0
    rel_sigma = np.sqrt(
        (edge_sd / radial) ** 2 + half_thr**2 + (0.5 * (speed_hi - speed_lo) / speed) ** 2
    )

    edge_times = np.sort(edges[:, 0])
    gaps = [
        (a, b) for a, b in zip(edge_times[:-1], edge_times[1:]) if b - a > 0.005 and a < t_peak < b
    ]
    rms = float(np.sqrt(np.mean((gompertz(edges[:, 0], *curve) - edges[:, 1]) ** 2)))
    return dict(
        speed=float(speed),
        sigma=float(rel_sigma * speed),
        speed_alpha_range=(float(speed_lo), float(speed_hi)),
        radial=radial,
        radial_alt_thr=float(radial_alt),
        t_peak_ms=1e3 * t_peak,
        r_peak_m=r_peak,
        los_depression_deg=depression,
        edge_sd_pct=100 * edge_sd / radial,
        thr_change_pct=(100 * (radial_alt / radial - 1) if np.isfinite(radial_alt) else None),
        edges_at_slope=not gaps,
        gap_ms=1e3 * (gaps[0][1] - gaps[0][0]) if gaps else 0.0,
        n_edges=int(len(edges)),
        rms_mm=1e3 * rms,
        curve=curve.tolist(),
        edges=edges.tolist(),
    )
