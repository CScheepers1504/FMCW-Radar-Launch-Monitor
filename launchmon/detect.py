"""Find the shot in a capture without a hand-set window.

    dmap = detection_map(prep.snr, prep.rbin)
    window = find_shot(dmap, prep.t, prep.rbin, BALLS["golf"])["window"]

Each frame is normalised, cells that stay lit (people, the net) are masked, and the best straight
line from 2.2 to 5.6 m at the ball's minimum speed or faster marks the shot.
"""

import warnings

import numpy as np

from .config import PARAMS


def frame_normalise(snr, rbin, band=None):
    lo, hi = band or PARAMS["norm_band_m"]
    k_lo, k_hi = int(lo / rbin), int(hi / rbin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are invalid frames
        median = np.nanmedian(snr[:, k_lo:k_hi], axis=1)
    return snr - np.nan_to_num(median)[:, None]


def occupancy_mask(snr, rbin):
    p = PARAMS
    lit = np.nan_to_num(snr, nan=0.0) > p["lit_db"]
    lit_count = np.cumsum(np.vstack([np.zeros((1, lit.shape[1])), lit]), axis=0)
    n = lit.shape[0]
    lo = np.clip(np.arange(n) - p["occ_half_w"], 0, n)
    hi = np.clip(np.arange(n) + p["occ_half_w"] + 1, 0, n)
    occupancy = (lit_count[hi] - lit_count[lo]) / (hi - lo)[:, None]
    mask = occupancy > p["occ_max"]
    k_lo, k_hi = int(1.0 / rbin), int(6.0 / rbin)
    interference = lit[:, k_lo:k_hi].mean(axis=1) > p["frame_lit_max"]
    mask[interference] = True
    masked = snr.copy()
    masked[mask & np.isfinite(snr)] = 0.0
    return masked, mask


def line_search(s, t, rbin, vmin=8.0, vmax=75.0, ra=2.2, rb=5.6):
    """Best straight line r = ra + v (t - t0) through the clipped map s, as (score, start frame, v)."""
    T = np.median(np.diff(t))
    n_frames = s.shape[0]
    best = (-1.0, 0, 0.0)
    for v in np.arange(vmin, vmax, 0.25):
        length = max(3, int(round((rb - ra) / v / T)))
        if length >= n_frames:
            continue
        bins = np.round((ra + v * np.arange(length) * T) / rbin).astype(int)
        starts = np.arange(n_frames - length)
        rows = starts[:, None] + np.arange(length)[None, :]
        score = np.max(np.stack([s[rows, bins + d] for d in (-1, 0, 1)]), 0).mean(1)
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), int(starts[i]), float(v))
    return best


def detection_map(snr, rbin):
    """The map searched for the shot: frame-normalised, then occupancy-masked."""
    return occupancy_mask(frame_normalise(snr, rbin), rbin)[0]


def find_shot(dmap, t, rbin, ball):
    """Best line over the whole take: its score, seed time and speed, and the window to track in."""
    clipped = np.clip(np.nan_to_num(dmap, nan=0.0), 0, 25)
    score, start, v = line_search(clipped, t, rbin, vmin=ball.v_min_seed)
    t_seed = float(t[start])
    return dict(
        score=round(score, 2),
        t_seed=round(t_seed, 3),
        v_seed=v,
        window=(round(t_seed, 3) - PARAMS["seed_before_s"], round(t_seed, 3) + ball.after_s),
    )
