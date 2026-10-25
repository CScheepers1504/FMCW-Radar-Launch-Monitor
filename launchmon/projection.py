"""Continue the fitted flight to landing with lift from an assumed spin.

    p = project(x, "golf", p0, radar, tau_last, spin_rpm=3000)    # p["carry_m"], p["apex_m"], ...

The path is measured up to the last radar frame and projected after it. Every assumed number is
in ASSUMED.
"""

import numpy as np

from .config import BALLS
from .flight import G, flight

RHO = 1.2  # air density, kg/m^3
MU = 1.81e-5  # air viscosity, Pa s

# Backspin presets. The "pro" rows are the TrackMan PGA Tour averages (US 2026/0115564 A1,
# Fig. 9: driver, 7-iron, pitching wedge). The other rows are not sourced. The Garmin R50 export
# gives 2250-4650 rpm (interquartile range) at 42-53 m/s ball speed, and the driver rows sit
# inside that band.
ASSUMED = dict(
    status="assumed spin; golf: Smits & Smith 1994; football/rugby: nominal, untested",
    spin_rpm={  # backspin, rpm
        "driver": {"pro": 2686, "intermediate": 3000, "beginner": 3500},
        "iron": {"pro": 7097, "intermediate": 6000, "beginner": 5000},  # 7-iron
        "wedge": {"pro": 9304, "intermediate": 7500, "beginner": 6000},  # pitching wedge
    },
    carry_band_rpm=(2250, 3440, 4650),  # golf carry band: R50 quartiles and median at 42-53 m/s
    lift=dict(slope=1.5, cap=0.35),  # football / rugby only: C_L = min(slope * S, cap), nominal
    spin_decay_s=25.0,  # football / rugby only: omega(t) = omega0 * exp(-t / tau)
)

# Golf aerodynamics, Smits & Smith (1994) wind-tunnel fits:
#   C_D = cd1 + cd2 W + cd3 sin(pi (Re - a1) / a2),  C_L = cl1 W^0.4,
#   d(omega)/dt = -r1 W V^2 / r^2,  with spin ratio W = omega r / V and Re = rho V d / mu.
# Stated for driver shots, Re 7e4-2.1e5 and W 0.08-0.20.
SMITS = dict(cd1=0.24, cd2=0.18, cd3=0.06, cl1=0.54, r1=2e-5, a1=9e4, a2=2e5)

# Radius m, mass kg. Golf and football are regulation values. The rugby ball is treated as a
# sphere of equal frontal area, which it is not, so its projection is indicative only.
BALL_PHYS = {"golf": (0.02135, 0.04593), "football": (0.11, 0.43), "rugby": (0.095, 0.44)}


def lift_coeff(S):
    """Nominal lift coefficient from the spin ratio S (football and rugby)."""
    lift = ASSUMED["lift"]
    return np.minimum(lift["slope"] * S, lift["cap"])


def flight_state(v0, el, az, k, tau, p0, h=1e-4):
    """Position and velocity of the fitted (drag + gravity) path at time tau after launch."""
    pos = flight(v0, el, az, k, np.array([tau - h, tau, tau + h]), p0)
    return pos[1], (pos[2] - pos[0]) / (2 * h)


def spin_axis(v, tilt_deg=0.0):
    """Unit backspin axis for a ball moving along v, tilted about v by tilt_deg (+ curves right)."""
    v_horizontal = np.array([v[0], v[1], 0.0])
    # horizontal, to the right of travel, so omega x v points up
    axis = np.cross(v_horizontal, [0, 0, 1.0])
    axis /= np.linalg.norm(axis)
    direction = v / np.linalg.norm(v)
    tilt = np.radians(tilt_deg)  # Rodrigues rotation of the axis about the direction
    return (
        axis * np.cos(tilt)
        + np.cross(direction, axis) * np.sin(tilt)
        + direction * np.dot(direction, axis) * (1 - np.cos(tilt))
    )


def simulate(pos0, vel0, ball, spin_rpm=0.0, tilt_deg=0.0, ground_z=0.0, t_max=15.0, dt=2e-3):
    """Drag + lift + gravity from (pos0, vel0) until the ball lands. Returns times and positions."""
    radius, mass = BALL_PHYS[ball]
    c0 = RHO * np.pi * radius**2 / (2 * mass)  # acceleration = c0 * coefficient * |v| * (...)
    k = BALLS[ball].k_drag  # = c0 * C_d, the same drag as the fit
    omega0 = spin_rpm * 2 * np.pi / 60
    axis = spin_axis(vel0, tilt_deg) if omega0 > 0 else np.zeros(3)
    decay_s = ASSUMED["spin_decay_s"]
    S = SMITS

    def acceleration(v, omega):
        """Acceleration and spin decay rate."""
        speed = np.linalg.norm(v)
        if ball == "golf":
            W = radius * omega / speed
            reynolds = RHO * speed * 2 * radius / MU
            cd = S["cd1"] + S["cd2"] * W + S["cd3"] * np.sin(np.pi * (reynolds - S["a1"]) / S["a2"])
            cl = S["cl1"] * W**0.4 if W > 0 else 0.0
            a = c0 * speed * (-cd * v + cl * np.cross(axis, v)) - np.array([0, 0, G])
            return a, -S["r1"] * omega * speed / radius
        a = -k * speed * v - np.array([0, 0, G])
        if omega0 > 0:
            a = a + c0 * lift_coeff(radius * omega / speed) * speed * np.cross(axis, v)
        return a, -omega / decay_s

    def derivative(state):
        a, d_omega = acceleration(state[3:6], state[6])
        return np.r_[state[3:6], a, d_omega]

    state = np.r_[pos0, vel0, omega0].astype(float)
    times, positions = [0.0], [state[:3].copy()]
    t = 0.0
    z_landed = ground_z + radius
    while t < t_max:
        k1 = derivative(state)
        k2 = derivative(state + dt / 2 * k1)
        k3 = derivative(state + dt / 2 * k2)
        k4 = derivative(state + dt * k3)
        new = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if new[2] <= z_landed and new[5] < 0:  # landed: interpolate the crossing
            frac = (state[2] - z_landed) / (state[2] - new[2])
            times.append(t + frac * dt)
            positions.append(state[:3] + frac * (new[:3] - state[:3]))
            break
        state, t = new, t + dt
        times.append(t)
        positions.append(state[:3].copy())
    return np.array(times), np.array(positions)


def project(x, ball, p0, radar, tau_last, spin_rpm=0.0, tilt_deg=0.0, tee_xy=(0, 0), n_meas=120):
    """Measured path to tau_last, then projected to landing. x = (v0, el, az, t_launch)."""
    v0, el, az, _ = x
    k = BALLS[ball].k_drag
    t_meas = np.linspace(0, tau_last, n_meas)
    measured = flight(v0, el, az, k, t_meas, p0) + radar
    pos, vel = flight_state(v0, el, az, k, tau_last, p0)
    t_proj, projected = simulate(pos + radar, vel, ball, spin_rpm, tilt_deg)
    t_proj = t_proj + tau_last
    landing = projected[-1]
    whole = np.r_[measured, projected]
    dx, dy = landing[0] - tee_xy[0], landing[1] - tee_xy[1]
    v_end = (projected[-1] - projected[-2]) / max(t_proj[-1] - t_proj[-2], 1e-9)
    return dict(
        t_meas=t_meas,
        meas=measured,
        t_proj=t_proj,
        proj=projected,
        carry_m=float(np.hypot(dx, dy)),
        carry_yd=float(np.hypot(dx, dy) / 0.9144),
        lateral_m=float(landing[1]),
        apex_m=float(whole[:, 2].max()),
        apex_x_m=float(whole[np.argmax(whole[:, 2]), 0]),
        flight_time_s=float(t_proj[-1]),
        landing=landing.tolist(),
        descent_deg=float(np.degrees(np.arctan2(-v_end[2], np.hypot(v_end[0], v_end[1])))),
        measured_to_x_m=float(measured[-1, 0]),
        spin_rpm=spin_rpm,
        tilt_deg=tilt_deg,
        assumed=ASSUMED["status"],
    )
