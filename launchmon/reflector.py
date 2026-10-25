"""Measure a corner reflector in a capture and the angle correction it implies.

result = measure(capture_dir, cal, xyz, session)    # angle errors, measured - surveyed
correction([result_1, result_2])                    # mean correction over the reflectors
"""

import json
import os

import numpy as np

from .capture import range_bin
from .geometry import AZ_SIGN, RX_X, TX_X, TX_Y, radar_dirs, u_from_rx, wrap_deg

# [tuned] reflector 33 dB when in the scene, 2-11 dB in the takes it was missing from
PRESENT_DB = 20.0
# [physical] about 2 range bins around the surveyed slant range, plus the static offset
RANGE_TOL_M = 0.10
# [measured] static range offset of the reflector's phase centre behind its apex
RANGE_OFFSET_M = 0.05
SEARCH_HALF_WIDTH_M = 0.2
MAX_FRAMES = 400
ORDER_FIT_MAX_DEG = 30.0  # the best slot rotation must fit the prediction this well (rms phase)
ORDER_MARGIN_MIN_DEG = 30.0  # and beat the next rotation by this much

# Which recorded slot holds each physical transmitter (Tx1, Tx2, Tx3), by calibration order name.
SLOT_ORDERS = {
    "as recorded": (0, 1, 2),
    "rotated one chirp": (2, 0, 1),
    "rotated two chirps": (1, 2, 0),
}


def _chirp_offset_int16(meta):
    """Samples to drop to reach a chirp block boundary, from the alignment stamp in meta.json."""
    stamp = meta.get("chirp_align")
    if stamp is None:
        return 0, "chirp alignment not stamped: assumed 0"
    if stamp.get("verdict") == "UNSURE":
        return 0, "chirp alignment UNSURE: assumed 0"
    return int(stamp["drop_int16"]), "chirp alignment %s" % stamp.get("verdict")


def _mean_spectrum(cap, meta, skip_int16, max_frames=MAX_FRAMES):
    """Range spectrum (slot x Rx x bin) averaged over loops and up to max_frames frames."""
    dims, cfg = meta["dims"], meta["config"]
    n_samples, n_rx, n_tx, n_loops = (
        dims["n_samples"],
        dims["n_rx"],
        dims["n_tx"],
        dims["n_loops"],
    )
    frame_int16 = dims["frame_bytes"] // 2
    if skip_int16 >= frame_int16:
        raise ValueError(
            "%s: alignment offset of %d int16 is a whole frame or more" % (cap, skip_int16)
        )
    with open(os.path.join(cap, "frames.bin"), "rb") as f:
        f.seek(2 * skip_int16)
        raw = np.fromfile(f, dtype=np.int16)
    n_frames = raw.size // frame_int16
    frames = np.unique(np.linspace(5, n_frames - 1, min(max_frames, n_frames - 5)).astype(int))
    window = np.hanning(n_samples + 2)[1:-1].astype(np.float32)
    swap_iq = cfg["adcbufCfg"]["SampleSwap"] == 1
    total = np.zeros((n_tx, n_rx, n_samples), np.complex128)
    for frame in frames:
        words = (
            raw[frame * frame_int16 : (frame + 1) * frame_int16].reshape(-1, 4).astype(np.float32)
        )
        iq = np.empty(words.shape[0] * 2, np.complex64)
        iq[0::2] = words[:, 0] + 1j * words[:, 2]
        iq[1::2] = words[:, 1] + 1j * words[:, 3]
        if swap_iq:
            iq = np.conj(iq)
        cube = iq.reshape(n_loops, n_tx, n_rx, n_samples)
        total += np.fft.fft(cube * window, axis=-1).mean(axis=0)
    return total / len(frames)


def _peak_bin(spectrum, rbin, r_lo, r_hi):
    """Strongest bin between r_lo and r_hi (power summed over channels), and its SNR over the median."""
    power = (np.abs(spectrum) ** 2).reshape(-1, spectrum.shape[-1]).sum(0)
    lo, hi = int(r_lo / rbin), int(np.ceil(r_hi / rbin))
    k = lo + int(np.argmax(power[lo : hi + 1]))
    return k, float(10 * np.log10(power[k] / np.median(power)))


def _predicted_phasors(u, w, rx_phase_deg, tx_phase_deg, el_sign, order):
    """Unit channel phasors (slot x Rx) a point target at direction (u, w) would give."""
    phasors = np.zeros((3, 4), complex)
    for tx in range(3):
        phase = (
            AZ_SIGN * np.pi * u * (TX_X[tx] + RX_X)
            + el_sign * np.pi * w * TX_Y[tx]
            + np.radians(rx_phase_deg + tx_phase_deg[tx])
        )
        phasors[order[tx]] = np.exp(1j * phase)
    return phasors


def _rms_phase_deg(measured, predicted):
    """rms phase distance between two channel sets after removing their common phase."""
    v = measured.ravel() / np.maximum(np.abs(measured.ravel()), 1e-30)
    p = predicted.ravel()
    diff = np.angle(v * np.conj(p) * np.exp(-1j * np.angle(np.vdot(p, v))))
    return float(np.degrees(np.sqrt(np.mean(diff**2))))


def analyse(cap, cal, xyz, radar_height_m, tilt_deg=0.0, yaw_deg=0.0):
    """One reflector at world position xyz: range, SNR, slot order and the angle errors."""
    with open(os.path.join(cap, "meta.json")) as f:
        meta = json.load(f)
    skip_int16, align_note = _chirp_offset_int16(meta)
    spectrum = _mean_spectrum(cap, meta, skip_int16)
    rbin = range_bin(meta)

    rel = np.asarray(xyz, float) - np.array([0.0, 0.0, radar_height_m])
    u_survey, w_survey = (float(v[0]) for v in radar_dirs(rel, tilt_deg, yaw_deg))
    slant = float(np.linalg.norm(rel))
    r_expected = slant + RANGE_OFFSET_M
    k, snr_db = _peak_bin(
        spectrum, rbin, r_expected - SEARCH_HALF_WIDTH_M, r_expected + SEARCH_HALF_WIDTH_M
    )
    channels = spectrum[:, :, k]

    rx_phase = np.array(cal["rx_phase_deg"], float)
    rx_scale = cal.get("rx_scale", 1.0)
    rx_scale = 1.0 if rx_scale is None or not np.isfinite(rx_scale) else float(rx_scale)
    delta, d13, el_sign = cal["delta_deg"], cal["d13_deg"], cal["el_scale_a"]
    tx_phase = np.array([0.0, delta + d13 / 2, d13])
    order = SLOT_ORDERS[cal["order_name"]]

    predicted = _predicted_phasors(
        u_survey * rx_scale, w_survey, rx_phase, tx_phase, el_sign, order
    )
    fit_deg = [_rms_phase_deg(np.roll(channels, -shift, axis=0), predicted) for shift in range(3)]
    shift = int(np.argmin(fit_deg))
    margin = float(sorted(fit_deg)[1] - fit_deg[shift])
    confident = fit_deg[shift] <= ORDER_FIT_MAX_DEG and margin >= ORDER_MARGIN_MIN_DEG

    aligned = np.roll(channels, -shift, axis=0)
    u_meas = u_from_rx(aligned, rx_phase) / rx_scale
    tx1, tx2, tx3 = order
    el_phase = np.angle(np.sum(aligned[tx2] ** 2 * np.conj(aligned[tx1]) * np.conj(aligned[tx3])))
    w_meas = float(
        np.angle(np.exp(1j * (el_phase - np.radians(2 * delta)))) / (2 * np.pi * el_sign)
    )
    delta_take = float(
        wrap_deg(np.degrees(el_phase - 2 * np.pi * el_sign * w_survey) / 2 - delta) + delta
    )

    def deg(x):
        return float(np.degrees(np.arcsin(np.clip(x, -1, 1))))

    chirp_int16 = meta["dims"]["n_samples"] * meta["dims"]["n_rx"] * 2
    return {
        "capture": os.path.basename(os.path.normpath(cap)),
        "range_expected": r_expected,
        "range_found": k * rbin,
        "range_offset": k * rbin - slant,
        "snr_db": snr_db,
        "chirp_align": align_note,
        "fit_rms_deg_per_k": fit_deg,
        "chirps_to_drop": shift,
        "margin_deg": margin,
        "verdict": "CONFIDENT" if confident else "UNSURE",
        "skip_int16": skip_int16 + shift * chirp_int16,
        "az_survey_deg": deg(u_survey),
        "az_meas_deg": deg(u_meas),
        "az_err_deg": deg(u_meas) - deg(u_survey),
        "el_survey_deg": deg(w_survey),
        "el_meas_deg": deg(w_meas),
        "el_err_deg": deg(w_meas) - deg(w_survey),
        "delta_take_deg": delta_take,
        "delta_cal_deg": delta,
    }


def measure(cap, cal, xyz, session):
    """analyse() plus whether the reflector is present and usable for the correction."""
    result = analyse(
        cap, cal, [float(v) for v in xyz], session.radar_height_m, session.tilt_deg, session.yaw_deg
    )
    result["range_err_m"] = float(result["range_found"] - result["range_expected"])
    result["present"] = bool(
        result["snr_db"] >= PRESENT_DB and abs(result["range_err_m"]) <= RANGE_TOL_M
    )
    result["usable"] = bool(result["present"] and result["verdict"] == "CONFIDENT")
    return result


def correction(results):
    """Mean (measured - surveyed) over the reflectors, with their spread when there are two or more."""
    if not results:
        return None
    az = np.array([r["az_err_deg"] for r in results])
    el = np.array([r["el_err_deg"] for r in results])
    spread = float(max(np.ptp(az), np.ptp(el))) if len(results) > 1 else None
    parts = ", ".join(
        "%s az %+.2f el %+.2f" % (r["capture"], r["az_err_deg"], r["el_err_deg"]) for r in results
    )
    return dict(
        az=float(az.mean()),
        el=float(el.mean()),
        n=len(results),
        spread=spread,
        source="mean of %d reflector(s), measured - surveyed: %s" % (len(results), parts),
    )
