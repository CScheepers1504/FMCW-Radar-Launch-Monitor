"""Session survey, ball constants and the fixed processing parameters.

Each parameter is tagged with its origin: [measured], [physical], [tuned 13 Oct] (chosen on the
13 Oct takes), [agreed 14 Oct] (acceptance rules fixed before tracking) or [assumed].
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

# Survey entries a session file must state. A missing one stops processing rather than
# silently taking the default below.
SURVEYED = ("radar_height_m", "tilt_deg", "yaw_deg", "tee_xy_m", "net_x_m")


@dataclass
class Session:
    """Everything measured once per setup. Loaded from JSON, with paths relative to the file."""

    name: str = ""
    radar_height_m: float = 0.52  # [measured] floor to array centre
    tilt_deg: float = -1.0  # [measured] boresight pitch, + up
    yaw_deg: float = 0.0  # [measured] boresight yaw, + left of target line
    tee_xy_m: tuple = (1.50, 0.0)  # [measured] tee downrange / lateral from the radar foot
    net_x_m: float = 6.00  # [measured] net plane downrange
    calibration: str = ""  # array phase calibration JSON
    # [measured] radar-frame elevations the calibration was validated over
    el_valid_span_deg: tuple = (-0.7, 10.8)
    # [measured] measured - surveyed
    angle_correction_deg: dict = field(default_factory=lambda: {"az": 0.0, "el": 0.0})
    angle_systematic_deg: float = 1.5  # [measured] bound on the correction (reflector disagreement)
    ball_centre_height_m: dict = field(default_factory=dict)  # [measured] per ball, from the floor
    # [measured] [{"xyz": [x, y, z] apex, "capture": dir}]
    reflectors: list = field(default_factory=list)
    notes: str = ""
    path: str = ""

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        missing = [k for k in SURVEYED if k not in data]
        if missing:
            raise ValueError(
                "%s: survey entries missing: %s (no fallback to another session)"
                % (path, ", ".join(missing))
            )
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "path"}
        session = cls(**data)
        session.path = os.path.abspath(path)
        if session.calibration and not os.path.isabs(session.calibration):
            session.calibration = os.path.normpath(
                os.path.join(os.path.dirname(session.path), session.calibration)
            )
        session.tee_xy_m = tuple(session.tee_xy_m)
        session.el_valid_span_deg = tuple(session.el_valid_span_deg)
        return session

    def save(self, path):
        data = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "path"}
        data["tee_xy_m"], data["el_valid_span_deg"] = (
            list(self.tee_xy_m),
            list(self.el_valid_span_deg),
        )
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    @property
    def radar(self):
        return np.array([0.0, 0.0, self.radar_height_m])

    def p0(self, ball):
        """Ball at launch relative to the array centre, world axes (x downrange, y left, z up)."""
        z = self.ball_centre_height_m.get(ball, BALLS[ball].centre_height_default)
        return np.array([self.tee_xy_m[0], self.tee_xy_m[1], z - self.radar_height_m])

    def load_cal(self):
        with open(self.calibration) as f:
            return json.load(f)


@dataclass(frozen=True)
class BallSpec:
    name: str
    k_drag: float  # [physical] rho Cd A / 2m, 1/m (rugby very uncertain)
    # range band fitted. Start [tuned 13 Oct] (club overlap below 2.5 m), end = net - 0.2
    band_m: tuple
    rms_limit_m: float  # [agreed 14 Oct] T1 acceptance limit on the range residual
    # [tuned 15 Sep] take the nearer local maximum (direct path, not floor bounce)
    nearest_peak: bool
    v_min_seed: float  # [physical] line-search minimum speed, m/s (club about 20 m/s radial)
    after_s: float  # [physical] detection window after the seed (flight time to the net)
    centre_height_default: float  # [measured 13 Oct] ball centre on its tee or the ground


BALLS = {
    "golf": BallSpec("golf", 4.84e-3, (2.4, 5.8), 0.015, False, 30.0, 0.30, 0.056),
    "football": BallSpec("football", 1.35e-2, (2.4, 5.8), 0.025, True, 6.0, 0.55, 0.11),
    "rugby": BallSpec(
        "rugby", 1.2 * 0.15 * 0.0284 / (2 * 0.44), (2.4, 5.8), 0.025, True, 6.0, 0.55, 0.25
    ),
}


def guess_ball(capture_dir):
    """Default for the GUI's ball selector, from the capture folder name."""
    name = os.path.basename(os.path.normpath(capture_dir)).lower()
    label = name[20:] if len(name) > 20 else name  # after the "YYYY-MM-DD_HH-MM-SS_" stamp
    for key, ball in (
        ("driver", "golf"),
        ("golf", "golf"),
        ("iron", "golf"),
        ("wedge", "golf"),
        ("football", "football"),
        ("rugby", "rugby"),
    ):
        if key in label:
            return ball
    return {"d": "golf", "f": "football", "r": "rugby"}.get(label[:1], "golf")


PARAMS = dict(
    # shot detection (detect.py)
    lit_db=6.0,  # [physical] a cell counts as lit above this SNR
    occ_half_w=50,  # [physical] +-150 ms occupancy window, frames
    occ_max=0.15,  # [physical] 3x the ball's own worst-case occupancy (5 of 101 frames)
    frame_lit_max=0.40,  # [physical] a frame this lit across 1-6 m is interference and is masked
    norm_band_m=(1.0, 6.0),
    seed_before_s=0.08,
    # range track (rangetrack.py)
    # [tuned 15 Sep] (gate half-width m, threshold dB)
    gate_passes=((0.30, 7.0), (0.12, 5.0), (0.07, 4.0)),
    sig_r_floor=3.5e-3,  # [physical] measured range-error floor, m
    min_points=6,  # [agreed 14 Oct] T1
    min_span_m=1.5,  # [agreed 14 Oct] T2
    max_sv_frac=0.02,  # [agreed 14 Oct] T2
    # Tx slot order (angles.py)
    slot_off_max_deg=30.0,
    slot_gap_min_deg=60.0,
    slot_r_min=0.8,  # [tuned 13 Oct]
    n_el_min=4,  # elevation counts as measured with >= 4 frames in the validated span
    # timing
    chirp_period_s=45e-6,  # [physical] profile idle + ramp time
    # club-head speed, golf only (club.py)
    club_map_s=(-0.050, 0.110),  # map window about the ball's launch, s
    club_window_s=(-0.045, 0.090),  # [tuned 13 Oct] fitted span, s
    club_gates_m=(0.35, 0.25, 0.20, 0.20),  # [tuned 13 Oct] gate half-widths per pass
    club_thr_db=4.0,  # [tuned 13 Oct] edge threshold
    club_thr_alt_db=6.0,  # alternative edge threshold, half its change enters the uncertainty
    club_min_edges=8,
    club_boot=300,  # edge resamples for the scatter
    club_head_height_m=0.05,  # [assumed] head centre above the ground at impact
    club_alpha_deg=5.0,  # [assumed] attack angle 0 +- this, not measured
)
