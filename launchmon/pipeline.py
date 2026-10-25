"""Run the whole chain on one capture.

    result = process(capture_dir, session, "golf")    # ShotResult
    print(result.summary())

Stages: prepare, detect, range, angles and net impact, then for golf the club speed and carry.
A rejected shot keeps whatever could be computed, with status REJECT and its reasons.
"""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import angles
from .capture import prepare
from .config import BALLS, PARAMS, Session
from .detect import detection_map, find_shot
from .rangetrack import fit_range

MS_TO_MPH = 2.23694


@dataclass
class ShotResult:
    capture: str
    ball: str
    status: str = "FAILED"  # ACCEPT / REJECT / FAILED
    reasons: list = field(default_factory=list)
    stage_failed: str = ""
    # headline: joint fit under the chosen shift
    speed: float = None  # m/s
    speed_sigma: float = None
    launch_el: float = None  # deg, world frame
    launch_el_sigma: float = None
    launch_el_sys: float = None
    launch_el_measured: bool = False  # >= 4 frames inside the calibrated elevation span
    launch_az: float = None  # deg, + left of the target line
    launch_az_sigma: float = None
    launch_az_sys: float = None
    t_launch: float = None  # s into the capture
    slot_shift: int = None
    slot_decided: bool = False
    net: dict = field(default_factory=dict)  # z, y (m), formal, systematic
    speed_range_only: float = None
    speed_range_only_sigma_pct: float = None
    # golf only
    club_speed: float = None  # m/s along the head's path, attack angle 0
    club_speed_sigma: float = None
    smash: float = None  # ball speed / club speed
    smash_sigma: float = None
    club: dict = field(default_factory=dict)  # radial speed, timing, uncertainty parts
    # carry in m at each spin of ASSUMED["carry_band_rpm"]
    carry: dict = field(default_factory=dict)
    # detail for the plots and the export
    detection: dict = field(default_factory=dict)
    range_row: dict = field(default_factory=dict)
    shifts: dict = field(default_factory=dict)
    prep_info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    _arrays: dict = field(default_factory=dict, repr=False)  # maps, tracks, frames (not exported)

    def summary(self):
        def fmt(value, spec):
            return (spec % value) if value is not None else "-"

        side = "" if self.launch_az is None else ("L" if self.launch_az > 0 else "R")
        lines = ["%s  %s  [%s]" % (os.path.basename(self.capture), self.ball, self.status)]
        if self.reasons:
            lines.append("  reasons: " + "; ".join(self.reasons))
        lines.append(
            "  ball speed   %s +- %s m/s  (%s mph)"
            % (
                fmt(self.speed, "%.2f"),
                fmt(self.speed_sigma, "%.2f"),
                fmt(self.speed and self.speed * MS_TO_MPH, "%.1f"),
            )
        )
        lines.append(
            "  launch angle %s +- %s (sys %s) deg%s"
            % (
                fmt(self.launch_el, "%.1f"),
                fmt(self.launch_el_sigma, "%.1f"),
                fmt(self.launch_el_sys, "%.1f"),
                "" if self.launch_el_measured else "  NOT MEASURED (range only)",
            )
        )
        lines.append(
            "  direction    %s%s +- %s (sys %s) deg"
            % (
                fmt(self.launch_az and abs(self.launch_az), "%.1f"),
                side,
                fmt(self.launch_az_sigma, "%.1f"),
                fmt(self.launch_az_sys, "%.1f"),
            )
        )
        if self.net:
            lines.append(
                "  at the net   height %s m, lateral %s m"
                % (fmt(self.net.get("z"), "%.2f"), fmt(self.net.get("y"), "%+.2f"))
            )
        if self.club_speed is not None:
            lines.append(
                "  club speed   %.1f +- %.1f m/s (radial %.1f, attack angle 0 +- %g deg assumed)%s"
                % (
                    self.club_speed,
                    self.club_speed_sigma,
                    self.club["radial"],
                    PARAMS["club_alpha_deg"],
                    (
                        ""
                        if self.club["edges_at_slope"]
                        else "  slope in a %.0f ms gap" % self.club["gap_ms"]
                    ),
                )
            )
            lines.append("  smash        %.2f +- %.2f" % (self.smash, self.smash_sigma))
        if self.carry:
            band = sorted(self.carry.items())
            lines.append(
                "  carry        %.0f-%.0f m at %d-%d rpm (projected, spin assumed)"
                % (band[0][1], band[-1][1], band[0][0], band[-1][0])
            )
        lines.append(
            "  Tx order     shift %s, %s"
            % (self.slot_shift, "decided" if self.slot_decided else "UNDECIDED")
        )
        return "\n".join(lines)

    def to_dict(self):
        def clean(obj):
            if isinstance(obj, dict):
                return {str(k): clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return clean(obj.tolist())
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            return obj

        return clean({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


def process(cap, session, ball, cache_dir=None, progress=None, window=None):
    """Run the chain on one capture. progress(stage, fraction) reports progress, and
    window=(t0, t1) replaces the automatic detection."""
    if isinstance(session, str):
        session = Session.load(session)
    spec = BALLS[ball]
    result = ShotResult(capture=os.path.abspath(cap), ball=ball)

    def stage_progress(stage):
        return (lambda fraction: progress(stage, fraction)) if progress else None

    clock = time.perf_counter

    start = clock()
    prep = prepare(cap, cache_dir, progress=stage_progress("prepare"))
    result.prep_info = prep.info
    result.timings["prepare"] = clock() - start

    start = clock()
    dmap = detection_map(prep.snr, prep.rbin)
    detection = find_shot(dmap, prep.t, prep.rbin, spec)
    if window is not None:
        detection.update(window=tuple(window), manual=True)
    result.detection = detection
    result.timings["detect"] = clock() - start
    if progress:
        progress("detect", 1.0)
    result._arrays.update(
        snr=prep.snr, raw=prep.raw_db, raw_pw=prep.raw_pw, dmap=dmap, t=prep.t, rbin=prep.rbin
    )

    start = clock()
    p0 = session.p0(ball)
    row, track = fit_range(prep, dmap, detection["window"], spec, p0)
    result.range_row = row
    result.status, result.reasons = row["status"], list(row["reasons"])
    result.timings["range"] = clock() - start
    if progress:
        progress("range", 1.0)
    if track is None or track.get("fit") is None:
        result.stage_failed = "range"
        if track is not None:
            result._arrays["track"] = track
        return result
    result._arrays["track"] = track
    result.speed_range_only, result.speed_range_only_sigma_pct = (row["v_geo"], row["sv_pct"])

    start = clock()
    angle_result = angles.analyse(
        prep, track, row, spec, session, session.load_cal(), progress=stage_progress("angles")
    )
    result.timings["angles"] = clock() - start
    shift = angle_result["chosen_shift"]
    chosen = angle_result["shifts"][shift]
    result.slot_shift, result.slot_decided = shift, angle_result["order_decided"]
    result.speed, result.speed_sigma = chosen["v"], chosen["sv"]
    result.launch_el, result.launch_el_sigma, result.launch_el_measured = (
        chosen["el"],
        chosen["sel"],
        chosen["el_measured"],
    )
    result.launch_az, result.launch_az_sigma = chosen["az"], chosen["saz"]
    result.t_launch = chosen["t_launch"]
    systematic = chosen.get("systematic", {})
    result.launch_el_sys, result.launch_az_sys = systematic.get("el"), systematic.get("az")
    result.net = dict(
        z=chosen["net"]["z"],
        y=chosen["net"]["y"],
        t=chosen["net"]["t"],
        z_formal=chosen["net_formal"]["z"],
        y_formal=chosen["net_formal"]["y"],
        z_sys=systematic.get("net_z"),
        y_sys=systematic.get("net_y"),
    )
    result.shifts = {s: dict(d) for s, d in angle_result["shifts"].items()}
    if not result.slot_decided:
        result.reasons.append(
            "Tx slot order undecided: elevation depends on it (all shifts reported)"
        )
    if not result.launch_el_measured:
        result.reasons.append(
            "elevation not measured (< %d frames in the calibrated span)" % PARAMS["n_el_min"]
        )
    result._arrays.update(
        frames=angle_result["frames"], frames_by_shift=angle_result["frames_by_shift"]
    )
    result._arrays["path"] = {
        s: angles.path(d["x"], spec.k_drag, p0, session.radar)
        for s, d in angle_result["shifts"].items()
    }
    result._arrays["curves"] = _curves(track, angle_result, spec, session, p0)
    if ball == "golf":
        _golf_extras(result, prep, chosen, angle_result, session, p0)
    return result


def _golf_extras(result, prep, chosen, angle_result, session, p0):
    """Club speed, smash factor and carry band. None of them feeds back into the launch state."""
    from .club import club_speed
    from .projection import ASSUMED, project

    start = time.perf_counter()
    x = chosen["x"]
    club = club_speed(
        prep, x[3], result._arrays["curves"][result.slot_shift], session.radar_height_m
    )
    if club is not None:
        result.club = {k: v for k, v in club.items() if k not in ("curve", "edges")}
        result._arrays["club"] = club
        result.club_speed, result.club_speed_sigma = club["speed"], club["sigma"]
        result.smash = result.speed / club["speed"]
        result.smash_sigma = result.smash * np.hypot(
            club["sigma"] / club["speed"], result.speed_sigma / result.speed
        )
    # last tracked frame, s after launch
    tau_last = float(np.max(angle_result["frames"]["t"]) - x[3])
    result.carry = {
        rpm: project(x, "golf", p0, session.radar, tau_last, rpm, 0.0, session.tee_xy_m)["carry_m"]
        for rpm in ASSUMED["carry_band_rpm"]
    }
    result.timings["golf_extras"] = time.perf_counter() - start


def _curves(track, angle_result, spec, session, p0):
    """Fitted range, elevation and azimuth curves for the plots."""
    from .flight import flight
    from .geometry import radar_dirs

    out = {}
    v0, el0, t_launch0 = track["fit"]
    t_obs = angle_result["frames"]["t"]
    tt = np.linspace(t_obs.min() - 0.02, t_obs.max() + 0.02, 200)
    for shift, fitted in angle_result["shifts"].items():
        x = fitted["x"]
        pos = flight(x[0], x[1], x[2], spec.k_drag, tt - x[3], p0)
        u, w = radar_dirs(pos, session.tilt_deg, session.yaw_deg)
        out[shift] = dict(
            t=tt,
            r=np.linalg.norm(pos, axis=1),
            el=np.degrees(np.arcsin(w)),
            az=np.degrees(np.arcsin(u)),
        )
    out["range_only"] = dict(
        t=tt, r=np.linalg.norm(flight(v0, el0, 0.0, spec.k_drag, tt - t_launch0, p0), axis=1)
    )
    return out
