"""Per-frame MIMO angles, Tx slot order, joint range + angle fit and the net impact.

    out = analyse(prep, track, range_row, ball, session, cal)

All three Tx slot shifts are fitted. The ball itself picks the shift: the Tx1-Tx3 phase must match
its fitted azimuth and range rate, and the shift counts as decided only if it wins clearly.
"""

import numpy as np

from .capture import read_frames
from .config import PARAMS
from .flight import flight, least_squares
from .geometry import AZ_SIGN, radar_dirs, to_world, u_from_rx, wrap_deg

LAM_SLOT = 299792458.0 / 78.79e9  # wavelength for the Doppler phase psi in the slot test


def _pick_bin(spectra, r, rbin, loops):
    """Of the bins either side of range r, the one with the most power."""
    k_centre = int(round(r / rbin))
    bins = [k for k in (k_centre - 1, k_centre, k_centre + 1) if 0 <= k < spectra.shape[-1]]
    return bins[int(np.argmax([np.sum(np.abs(spectra[loops, ..., k]) ** 2) for k in bins]))]


def frame_measurements(rp, rr, rbin, loops, cal, psi):
    """Per frame: radar-frame elevation and azimuth (deg, uncorrected) and the slot-test phase
    residual (rad)."""
    rx_phase = np.array(cal["rx_phase_deg"])
    rx_scale = cal.get("rx_scale") or 1.0
    el_sign, delta, d13 = cal["el_scale_a"], cal["delta_deg"], cal["d13_deg"]
    el, az, slot_residual = [], [], []
    for i, r in enumerate(rr):
        k = _pick_bin(rp[i], r, rbin, loops)
        V = rp[i, loops, :, :, k]  # loops x slots x Rx at the ball's bin
        el_phase = np.angle(np.sum(V[:, 1] ** 2 * np.conj(V[:, 0]) * np.conj(V[:, 2])))
        w = np.angle(np.exp(1j * (el_phase - np.radians(2 * delta)))) / (2 * np.pi * el_sign)
        u = u_from_rx(V.reshape(-1, V.shape[-1]), rx_phase)
        el.append(np.degrees(np.arcsin(np.clip(w, -1, 1))))
        az.append(np.degrees(np.arcsin(np.clip(u / rx_scale, -1, 1))))
        V_corrected = V * np.exp(-1j * np.radians(rx_phase))
        tx13_phase = np.angle(np.sum(V_corrected[:, 2] * np.conj(V_corrected[:, 0])))
        slot_residual.append(tx13_phase - (AZ_SIGN * np.pi * 4 * u + 2 * psi[i] + np.radians(d13)))
    return np.array(el), np.array(az), np.array(slot_residual)


def joint_fit(tf, rr, el_m, az_m, use_el, k, sig_r, x0, p0, tilt, yaw):
    """Fit (v0, el, az, t_launch) to the ranges and the per-frame angles together."""
    u_meas, w_meas = np.sin(np.radians(az_m)), np.sin(np.radians(el_m))
    use_az = np.ones(tf.size, bool)
    sigma_w, sigma_u = np.radians(2.0), np.radians(2.0)
    for _ in range(4):

        def residuals(params):
            v0, el, az, t_launch = params
            pos = flight(v0, el, az, k, tf - t_launch, p0)
            u, w = radar_dirs(pos, tilt, yaw)
            return np.r_[
                (np.linalg.norm(pos, axis=1) - rr) / sig_r,
                (w_meas - w)[use_el] / sigma_w,
                (u_meas - u)[use_az] / sigma_u,
            ]

        solution = least_squares(residuals, x0, x_scale=[10, 0.1, 0.1, 0.01])
        x0 = solution.x
        pos = flight(*solution.x[:3], k, tf - solution.x[3], p0)
        u, w = radar_dirs(pos, tilt, yaw)
        error_w, error_u = w_meas - w, u_meas - u
        mad_w = 1.4826 * np.median(np.abs(error_w[use_el])) if use_el.any() else sigma_w
        mad_u = 1.4826 * np.median(np.abs(error_u[use_az]))
        sigma_w, sigma_u = max(mad_w, np.radians(0.3)), max(mad_u, np.radians(0.3))
        use_el = use_el & (np.abs(error_w) < 3 * sigma_w)
        use_az = np.abs(error_u) < 3 * sigma_u
    cov = np.linalg.pinv(solution.jac.T @ solution.jac)
    chi2 = float(np.mean(solution.fun**2))
    return (
        solution.x,
        np.sqrt(np.diag(cov) * max(chi2, 1.0)),
        dict(
            sig_el_frame=float(np.degrees(sigma_w)),
            sig_az_frame=float(np.degrees(sigma_u)),
            n_el=int(use_el.sum()),
            n_az=int(use_az.sum()),
            chi2=chi2,
            use_el=use_el.copy(),
            use_az=use_az.copy(),
        ),
    )


def path(x, k, p0, radar, t_end=1.5, n=3001):
    """Times and world positions of the fitted path from launch to t_end."""
    v0, el, az, _ = x
    tt = np.linspace(0, t_end, n)
    return tt, flight(v0, el, az, k, tt, p0) + radar


def net_hit(x, k, p0, radar, net_x):
    """Time, height and lateral position where the fitted path crosses the net plane."""
    tt, pos = path(x, k, p0, radar)
    i = int(np.argmax(pos[:, 0] >= net_x))
    if i == 0:
        return dict(t=None, z=None, y=None)
    frac = (net_x - pos[i - 1, 0]) / (pos[i, 0] - pos[i - 1, 0])
    crossing = pos[i - 1] + frac * (pos[i] - pos[i - 1])
    return dict(
        t=float(tt[i - 1] + frac * (tt[i] - tt[i - 1])), z=float(crossing[2]), y=float(crossing[1])
    )


def analyse(prep, track, range_row, ball, session, cal, progress=None):
    """Fit all three slot shifts, choose one, and compute the net impact with its errors."""
    fr, rr = track["fr"], track["rr"]
    v0, el0, t_launch0 = track["fit"]
    t = prep.t
    tf = t[fr]
    k = ball.k_drag
    p0, radar = session.p0(ball.name), session.radar
    tilt, yaw = session.tilt_deg, session.yaw_deg
    correction = session.angle_correction_deg
    el_span = session.el_valid_span_deg
    sig_r = max(range_row["rms_mm"] * 1e-3, PARAMS["sig_r_floor"])

    h = 1e-4
    range_rate = (
        np.linalg.norm(flight(v0, el0, 0, k, tf - t_launch0 + h, p0), axis=1)
        - np.linalg.norm(flight(v0, el0, 0, k, tf - t_launch0 - h, p0), axis=1)
    ) / (2 * h)
    psi = 4 * np.pi * range_rate * PARAMS["chirp_period_s"] / LAM_SLOT

    n_frames = len(t)
    shifts = {}
    for shift in range(3):
        spectra = read_frames(prep.cap, prep.meta, prep.skip_int16, fr, shift)
        if shift == 0:
            spectra -= prep.mean[None]
        else:  # a shifted read needs its own clutter mean
            sample = np.linspace(5, n_frames - 2, 300).astype(int)
            spectra -= read_frames(prep.cap, prep.meta, prep.skip_int16, sample, shift).mean(
                axis=0
            )[None]
        # shifted, the last triple straddles the frame gap
        loops = slice(None) if shift == 0 else slice(0, 3)
        el_m, az_m, slot_residual = frame_measurements(spectra, rr, prep.rbin, loops, cal, psi)
        el_c, az_c = el_m - correction["el"], az_m - correction["az"]
        use_el = (el_c >= el_span[0]) & (el_c <= el_span[1])
        x, sig, info = joint_fit(
            tf,
            rr,
            el_c,
            az_c,
            use_el,
            k,
            sig_r,
            [v0, np.radians(12.0), 0.0, t_launch0],
            p0,
            tilt,
            yaw,
        )
        mean_phasor = np.mean(np.exp(1j * slot_residual))
        shifts[shift] = dict(
            el_frames=el_c,
            az_frames=az_c,
            x=x,
            sig=sig,
            info=info,
            slot_R=float(np.abs(mean_phasor)),
            slot_offset_deg=float(np.degrees(np.angle(mean_phasor))),
        )
        if progress:
            progress((shift + 1) / 4)

    offset = {s: abs(float(wrap_deg(shifts[s]["slot_offset_deg"]))) for s in shifts}
    best = min(offset, key=offset.get)
    second = sorted(offset.values())[1]
    decided = (
        offset[best] <= PARAMS["slot_off_max_deg"]
        and second - offset[best] >= PARAMS["slot_gap_min_deg"]
        and shifts[best]["slot_R"] >= PARAMS["slot_r_min"]
    )

    out = dict(chosen_shift=best, order_decided=bool(decided), shifts={})
    for shift, fitted in shifts.items():
        x, sig, info = fitted["x"], fitted["sig"], fitted["info"]
        hit = net_hit(x, k, p0, radar, session.net_x_m)
        x_el_up, x_az_up = np.array(x, float), np.array(x, float)
        x_el_up[1] += sig[1]
        x_az_up[2] += sig[2]
        hit_el = net_hit(x_el_up, k, p0, radar, session.net_x_m)
        hit_az = net_hit(x_az_up, k, p0, radar, session.net_x_m)
        out["shifts"][shift] = dict(
            v=float(x[0]),
            sv=float(sig[0]),
            el=float(np.degrees(x[1])),
            sel=float(np.degrees(sig[1])),
            az=float(np.degrees(x[2])),
            saz=float(np.degrees(sig[2])),
            t_launch=float(x[3]),
            n_el=info["n_el"],
            el_measured=info["n_el"] >= PARAMS["n_el_min"],
            sig_el_frame=info["sig_el_frame"],
            sig_az_frame=info["sig_az_frame"],
            chi2=info["chi2"],
            slot_R=fitted["slot_R"],
            slot_offset_deg=fitted["slot_offset_deg"],
            net=hit,
            net_formal=dict(
                z=(abs(hit_el["z"] - hit["z"]) if None not in (hit_el["z"], hit["z"]) else None),
                y=(abs(hit_az["y"] - hit["y"]) if None not in (hit_az["y"], hit["y"]) else None),
            ),
            x=list(map(float, x)),
        )

    # Systematic error, on the chosen shift only (4 refits).
    chosen = shifts[best]
    hit = out["shifts"][best]["net"]
    sys_z, sys_y, sys_el, sys_az = [], [], [], []
    if hit["z"] is not None:
        bound = session.angle_systematic_deg
        for d_el, d_az in ((bound, 0), (-bound, 0), (0, bound), (0, -bound)):
            x_sys, _, _ = joint_fit(
                tf,
                rr,
                chosen["el_frames"] + d_el,
                chosen["az_frames"] + d_az,
                chosen["info"]["use_el"],
                k,
                sig_r,
                list(chosen["x"]),
                p0,
                tilt,
                yaw,
            )
            hit_sys = net_hit(x_sys, k, p0, radar, session.net_x_m)
            if d_el:
                sys_el.append(abs(np.degrees(x_sys[1] - chosen["x"][1])))
                if hit_sys["z"] is not None:
                    sys_z.append(abs(hit_sys["z"] - hit["z"]))
            else:
                sys_az.append(abs(np.degrees(x_sys[2] - chosen["x"][2])))
                if hit_sys["y"] is not None:
                    sys_y.append(abs(hit_sys["y"] - hit["y"]))
    out["shifts"][best]["systematic"] = dict(
        el=max(sys_el) if sys_el else None,
        az=max(sys_az) if sys_az else None,
        net_z=max(sys_z) if sys_z else None,
        net_y=max(sys_y) if sys_y else None,
    )
    if progress:
        progress(1.0)

    # Per-frame 3D points under the chosen shift.
    points = to_world(rr, chosen["el_frames"], chosen["az_frames"], radar, tilt, yaw)
    out["frames"] = dict(
        t=tf,
        r=rr,
        el=chosen["el_frames"],
        az=chosen["az_frames"],
        xyz=points,
        el_used=chosen["info"]["use_el"],
        az_used=chosen["info"]["use_az"],
        sig_lat_m=rr * np.radians(chosen["info"]["sig_az_frame"]),
        sig_vert_m=rr * np.radians(chosen["info"]["sig_el_frame"]),
    )
    out["frames_by_shift"] = {
        s: dict(el=f["el_frames"], az=f["az_frames"], use_el=f["info"]["use_el"])
        for s, f in shifts.items()
    }
    return out
