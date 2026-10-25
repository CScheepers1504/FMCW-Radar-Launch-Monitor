"""Array geometry, direction estimates and conversion between the world and radar frames.

World frame: x downrange, y left, z up, origin at the radar's foot.
Radar frame: u = sin(az) to the left, w = sin(el) up. Antenna positions in half wavelengths.
"""

import numpy as np

AZ_SIGN = -1  # the FFT kernel maps positive spatial frequency to a negative bin
RX_X = np.arange(4)
TX_X = np.array([0, 2, 4])
TX_Y = np.array([0, 1, 0])  # Tx2 is raised by half a wavelength: the elevation baseline
C = 299792458.0


def rx_vec(V):
    """Per-Rx phasor relative to Rx0, summed coherently over the Tx slots, so it does not
    depend on the slot order. V is slots x Rx."""
    return np.array([np.sum(V[:, rx] * np.conj(V[:, 0])) for rx in range(V.shape[1])])


def u_from_rx(V, rho_deg=None, grid=np.linspace(-0.7, 0.7, 2801)):
    """Matched-filter azimuth direction cosine u (to the left) from the 4 Rx, after removing
    the calibrated Rx phase offsets rho."""
    phasor = rx_vec(V)
    if rho_deg is not None:
        phasor = phasor * np.exp(-1j * np.radians(rho_deg))
    response = np.abs(np.exp(-1j * AZ_SIGN * np.pi * np.outer(grid, RX_X)) @ phasor)
    return float(grid[int(np.argmax(response))])


def _rotation(tilt_deg, yaw_deg):
    """World axes to radar axes: undo the yaw, then the tilt."""
    tilt, yaw = np.radians(tilt_deg), np.radians(yaw_deg)
    undo_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    undo_tilt = np.array(
        [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]]
    )
    return undo_tilt @ undo_yaw


def radar_dirs(rel, tilt_deg=0.0, yaw_deg=0.0):
    """(u, w) of points given relative to the array centre in world axes, one point per row."""
    q = np.atleast_2d(rel) @ _rotation(tilt_deg, yaw_deg).T
    r = np.linalg.norm(q, axis=1)
    return q[:, 1] / r, q[:, 2] / r


def to_world(r, el_deg, az_deg, radar_xyz, tilt_deg=0.0, yaw_deg=0.0):
    """Range and radar-frame angles to world points (the inverse of radar_dirs)."""
    u, w = np.sin(np.radians(az_deg)), np.sin(np.radians(el_deg))
    q = np.stack([np.sqrt(np.clip(1 - u**2 - w**2, 0, 1)), u, w], axis=1) * np.asarray(r)[:, None]
    return q @ _rotation(tilt_deg, yaw_deg) + np.asarray(radar_xyz)


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0
