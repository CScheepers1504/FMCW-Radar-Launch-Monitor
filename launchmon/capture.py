"""Read a capture folder and build its range-time SNR map.

    prep = prepare(capture_dir)      # realigned and cached; prep.snr is frames x range bins, dB

A capture folder holds frames.bin, index.csv and meta.json, as written by the recorder.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import C

NOT_TIMED = -2147483648  # burst phase of a frame start the data node could not time
CONTRAST_MIN, MARGIN_MIN = (2.0, 1.3)  # sub-chirp boundary thresholds, as in chirp_align_check.py
CACHE_VERSION = "3"


def is_capture(path):
    return all(
        os.path.isfile(os.path.join(path, f)) for f in ("frames.bin", "index.csv", "meta.json")
    )


def read_meta(cap):
    with open(os.path.join(cap, "meta.json")) as f:
        return json.load(f)


def read_index(cap, frame_bytes=None):
    # Sequence number, stream offset and burst phase per frame.

    table = np.genfromtxt(
        os.path.join(cap, "index.csv"), delimiter=",", names=True, dtype=None, encoding=None
    )
    seq = np.atleast_1d(table["seq"]).astype(np.int64)
    if "stream_offset" in table.dtype.names:
        return (seq, table["stream_offset"].astype(np.int64), table["burst_phase"].astype(np.int64))
    return (seq, (seq - seq[0]) * int(frame_bytes), np.full(seq.shape, NOT_TIMED, np.int64))


def is_legacy(cap):
    with open(os.path.join(cap, "index.csv")) as f:
        return "stream_offset" not in f.readline()


def _unpack_iq(words):
    """AWR1843 two-lane packing: each group of four int16 is [I0 I1 Q0 Q1], two complex samples."""
    iq = np.empty(2 * words.shape[0], np.complex64)
    iq[0::2] = words[:, 0] + 1j * words[:, 2]
    iq[1::2] = words[:, 1] + 1j * words[:, 3]
    return iq


# Sub-chirp alignment: the sample-to-sample jump is largest at an Rx-block boundary.
def block_profile(raw, ns, phase):
    """Mean |x[n+1] - x[n]| at each position within an Rx block of ns samples, for one int16
    packing phase."""
    words = raw[phase:]
    words = words[: (len(words) // 4) * 4].reshape(-1, 4)
    iq = _unpack_iq(words)
    iq = iq[: (len(iq) // ns) * ns]
    jump = np.abs(np.diff(iq))
    position = np.arange(len(jump)) % ns
    return np.bincount(position, weights=jump, minlength=ns) / np.bincount(position, minlength=ns)


def chirp_align(cap, meta, n_frames=200, skip_frames=5):
    """Find the Rx-block boundary in the samples and the int16 to drop to reach it."""
    dims = meta["dims"]
    n_samples, frame_int16 = dims["n_samples"], dims["frame_bytes"] // 2
    path = os.path.join(cap, "frames.bin")
    n = min(n_frames, os.path.getsize(path) // 2 // frame_int16 - skip_frames)
    with open(path, "rb") as f:
        f.seek(skip_frames * frame_int16 * 2)
        raw = np.fromfile(f, dtype=np.int16, count=n * frame_int16).astype(np.float32)
    best = None
    for phase in range(4):
        profile = block_profile(raw, n_samples, phase)
        ranked = np.argsort(profile)[::-1]
        contrast = profile[ranked[0]] / max(np.median(profile), 1e-9)
        if best is None or contrast > best["contrast"]:
            best = dict(
                phase=phase,
                pos=int(ranked[0]),
                contrast=float(contrast),
                margin=float(profile[ranked[0]] / max(profile[ranked[1]], 1e-9)),
            )
    into_block = (n_samples - 1 - best["pos"]) % n_samples
    best["drop_int16"] = int(
        (best["phase"] + 2 * ((n_samples - into_block) % n_samples)) % (2 * n_samples)
    )
    return best


def realign(cap, meta):
    """Chirp and sub-chirp alignment, true frame numbers and the invalid-frame mask."""
    dims = meta["dims"]
    chirp_bytes = dims["n_samples"] * dims["n_rx"] * dims["iq"] * dims["bytes_per_sample"]
    chirps_per_frame = dims["n_loops"] * dims["n_tx"]
    seq, stream_offset, burst_phase = read_index(cap, dims["frame_bytes"])
    timed = burst_phase != NOT_TIMED
    offset_chirps = np.full(burst_phase.shape, np.nan)
    offset_chirps[timed] = np.round(burst_phase[timed] / chirp_bytes)
    steady = int(np.median(offset_chirps[timed])) if timed.any() else 0
    skip_chirps = steady % chirps_per_frame

    stamp = meta.get("chirp_align")
    if stamp and stamp.get("verdict") in ("ALIGNED", "OFFSET"):
        sub_chirp_drop, sub_chirp_source = (int(stamp["drop_int16"]), "stamp %s" % stamp["verdict"])
    else:
        found = chirp_align(cap, meta)
        ok = found["contrast"] >= CONTRAST_MIN and found["margin"] >= MARGIN_MIN
        if not ok:
            # A strong static return raises the detector floor: accept phase 0 when its own peak
            # sits at the block boundary and clears the thresholds.
            raw = np.fromfile(
                os.path.join(cap, "frames.bin"),
                dtype=np.int16,
                count=205 * dims["frame_bytes"] // 2,
            )[5 * dims["frame_bytes"] // 2 :].astype(np.float32)
            profile = block_profile(raw, dims["n_samples"], 0)
            ranked = np.argsort(profile)[::-1]
            contrast = profile[ranked[0]] / np.median(profile)
            margin = profile[ranked[0]] / profile[ranked[1]]
            if (
                ranked[0] == dims["n_samples"] - 1
                and contrast >= CONTRAST_MIN
                and margin >= MARGIN_MIN
            ):
                found, ok = {"drop_int16": 0, "contrast": contrast, "margin": margin}, True
        sub_chirp_drop = int(found["drop_int16"]) if ok else 0
        sub_chirp_source = "measured %s (contrast %.1f, margin %.2f)" % (
            "ok" if ok else "UNSURE->0",
            found["contrast"],
            found["margin"],
        )

    n = stream_offset.size
    frame_number = (stream_offset - stream_offset[0]) // dims["frame_bytes"]
    valid = np.ones(n, bool)
    valid[timed] &= offset_chirps[timed] == steady
    if skip_chirps:
        # skipping chirps moves each frame's tail into the next frame, so it needs that frame too
        valid &= np.r_[frame_number[1:] - frame_number[:-1] == 1, False]
    return dict(
        skip_chirps=int(skip_chirps),
        steady_offset_chirps=steady,
        ca_drop_int16=sub_chirp_drop,
        ca_src=sub_chirp_source,
        skip_int16=int(sub_chirp_drop + skip_chirps * chirp_bytes // 2),
        fidx=frame_number,
        valid=valid,
        frames_missed=int(frame_number[-1] + 1 - n),
        off_grid=int((timed & (offset_chirps != steady)).sum()),
    )


# Range spectra
def hann_unit():
    """Hann window factory, normalised to unit rms so SNR does not depend on the window."""

    def window(ns):
        x = np.hanning(ns + 2)[1:-1].astype(np.float32)
        return x / np.sqrt(np.mean(x**2))

    return window


def _decode(raw_block, n_loops, n_tx, n_rx, n_samples, swap_iq, window):
    """Raw int16 of whole frames to range spectra (frames x loops x slots x Rx x bins)."""
    iq = _unpack_iq(np.asarray(raw_block).reshape(-1, 4).astype(np.float32))
    if swap_iq:
        iq = np.conj(iq)
    return np.fft.fft(iq.reshape(-1, n_loops, n_tx, n_rx, n_samples) * window, axis=-1).astype(
        np.complex64
    )


def range_bin(meta):
    """Range bin width, m."""
    profile = meta["config"]["profileCfg"]
    return (
        profile["digOutSampleRate"]
        * 1e3
        * C
        / (2 * profile["freqSlopeConst"] * 1e12)
        / meta["dims"]["n_samples"]
    )


def read_frames(cap, meta, skip_int16, frames, extra_chirps=0):
    """Range spectra (frames x loops x slots x Rx x bins) of the given frames, optionally read
    extra_chirps late to test another Tx slot order."""
    dims, cfg = meta["dims"], meta["config"]
    n_samples, n_rx, n_tx, n_loops = (
        dims["n_samples"],
        dims["n_rx"],
        dims["n_tx"],
        dims["n_loops"],
    )
    frame_int16 = dims["frame_bytes"] // 2
    raw = np.memmap(os.path.join(cap, "frames.bin"), dtype=np.int16, mode="r")
    start = skip_int16 + extra_chirps * n_samples * n_rx * 2
    window = hann_unit()(n_samples)
    swap_iq = cfg["adcbufCfg"]["SampleSwap"] == 1
    out = np.empty((len(frames), n_loops, n_tx, n_rx, n_samples), np.complex64)
    for i, frame in enumerate(frames):
        block = raw[start + frame * frame_int16 : start + (frame + 1) * frame_int16]
        out[i] = _decode(block, n_loops, n_tx, n_rx, n_samples, swap_iq, window)[0]
    return out


@dataclass
class Prepared:
    cap: str
    meta: dict
    # frames x bins, dB over the per-bin median, clutter mean removed; NaN = invalid
    snr: np.ndarray
    t: np.ndarray  # s, from the stream offset (true time through dropped frames)
    valid: np.ndarray
    rbin: float  # range bin width, m
    T: float  # frame period, s
    mean: np.ndarray  # complex clutter mean, loops x slots x Rx x bins
    skip_int16: int
    info: dict = field(default_factory=dict)
    raw_db: np.ndarray = None  # frames x bins, dB over the same per-bin floor, clutter NOT removed
    # frames x bins, dB re 1 ADC count^2, no normalisation, clutter NOT removed
    raw_pw: np.ndarray = None

    @property
    def first_after_boot(self):
        """H1: the 282 Hz first-after-boot mode, which leaves most frames invalid."""
        if self.info.get("legacy_index"):
            rate = self.info.get("publisher_rate_hz") or 333.0
            return rate < 300.0
        return self.info["invalid"] >= 0.5 * self.info["frames"]


def _cache_key(cap):
    stat = os.stat(os.path.join(cap, "frames.bin"))
    key = "%s|%d|%d|%s" % (os.path.abspath(cap), stat.st_size, int(stat.st_mtime), CACHE_VERSION)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def default_cache_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "launchmon_cache")


def prepare(cap, cache_dir=None, progress=None, chunk=400):
    """Capture folder -> Prepared, cached by the path, size and age of frames.bin."""
    meta = read_meta(cap)
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _cache_key(cap) + ".npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return Prepared(
            cap,
            meta,
            z["snr"],
            z["t"],
            z["valid"],
            float(z["rbin"]),
            float(z["T"]),
            z["mean"],
            int(z["skip_int16"]),
            info,
            z["raw_db"],
            z["raw_pw"],
        )

    alignment = realign(cap, meta)
    dims, cfg = meta["dims"], meta["config"]
    n_samples, n_rx, n_tx, n_loops = (
        dims["n_samples"],
        dims["n_rx"],
        dims["n_tx"],
        dims["n_loops"],
    )
    frame_int16 = dims["frame_bytes"] // 2
    raw = np.memmap(os.path.join(cap, "frames.bin"), dtype=np.int16, mode="r")[
        alignment["skip_int16"] :
    ]
    n_frames = raw.size // frame_int16
    window = hann_unit()(n_samples)
    swap_iq = cfg["adcbufCfg"]["SampleSwap"] == 1
    spectra = np.empty((n_frames, n_loops, n_tx, n_rx, n_samples), np.complex64)
    for first in range(0, n_frames, chunk):
        last = min(n_frames, first + chunk)
        spectra[first:last] = _decode(
            raw[first * frame_int16 : last * frame_int16],
            n_loops,
            n_tx,
            n_rx,
            n_samples,
            swap_iq,
            window,
        )
        if progress:
            progress(last / n_frames)
    rbin = range_bin(meta)
    T = cfg["frameCfg"]["framePeriod"] * 1e-3
    valid = alignment["valid"][:n_frames]
    t = alignment["fidx"][:n_frames] * T
    clutter = spectra[valid].mean(axis=0)
    power_raw = (np.abs(spectra) ** 2).sum(axis=(1, 2, 3))
    spectra -= clutter
    power = (np.abs(spectra) ** 2).sum(axis=(1, 2, 3))
    del spectra
    noise = np.median(power[valid], axis=0)
    snr = (10 * np.log10(power / noise)).astype(np.float32)
    snr[~valid] = np.nan
    raw_db = (10 * np.log10(power_raw / noise)).astype(np.float32)
    raw_db[~valid] = np.nan
    raw_pw = (10 * np.log10(power_raw)).astype(np.float32)
    raw_pw[~valid] = np.nan
    info = {k: v for k, v in alignment.items() if k not in ("fidx", "valid")}
    info.update(
        frames=int(n_frames),
        invalid=int((~valid).sum()),
        legacy_index=is_legacy(cap),
        publisher_rate_hz=meta.get("capture", {}).get("publisher_rate_hz"),
        recorded_utc=meta.get("recorded_utc"),
    )
    np.savez_compressed(
        cache_path,
        snr=snr,
        t=t,
        valid=valid,
        rbin=rbin,
        T=T,
        mean=clutter.astype(np.complex64),
        skip_int16=alignment["skip_int16"],
        info=json.dumps(info),
        raw_db=raw_db,
        raw_pw=raw_pw,
    )
    return Prepared(
        cap,
        meta,
        snr,
        t,
        valid,
        rbin,
        T,
        clutter.astype(np.complex64),
        alignment["skip_int16"],
        info,
        raw_db,
        raw_pw,
    )
