# launchmon: capture folder → launch metrics
This code should be generalisable for different FMCW radar systems, but this has not been tested.

The analysis pipeline for a TI AWR1843Boost + DCA1000EVM FMCW radar, purposed as a general sports ball launch monitor. This code is one repository of two that were used for collecting and processing, developed at the University of Cape Town, 2026. 

It turns one capture from the [AWR1843 capture driver](https://github.com/CScheepers1504/AWR1843-Capture-Driver) into a ball's launch speed, elevation and
azimuth, with their uncertainties, and for golf the club-head speed, smash factor and a projected carry band.

For the full breakdown of the approach, design and results, please refer to the `.PDF` within the repo.

```
python -m launchmon                                   # GUI
python -m launchmon <capture_dir> --ball golf         # command line, prints the metrics
python -m launchmon <capture_dir> --json out.json     # ... and saves everything
```

Run from the folder that contains `launchmon/`. Needs Python 3.9+, numpy, and for the GUI PyQt6,
pyqtgraph and PyOpenGL (Pillow for GIF export).

## Inputs

| Input | Where | Per |
|---|---|---|
| capture folder (`frames.bin`, `index.csv`, `meta.json`) | recorder output | shot |
| ball type (golf / football / rugby) | GUI dropdown | shot |
| session file: radar height, tilt, yaw, tee, net, calibration file, angle correction, ball heights | `sessions/*.json` | setup |
| array phase calibration | `calibration/*.json`, named in the session file | array |

## Chain

| Stage | Module |
|---|---|
| realign, frame times, range-time SNR map (cached) | `capture.py` |
| automatic shot detection with no hand-set window | `detect.py` |
| range track,  drag + gravity fit, acceptance tests H1 H2 T1 T2 | `rangetrack.py` |
| MIMO angles per Tx slot shift, slot-order test, joint fit, net impact | `angles.py` |
| club-head speed and smash factor, golf only | `club.py` |
| projection to landing from an assumed spin | `projection.py` |
| flight model and solvers | `flight.py` |
| array geometry and frames | `geometry.py` |
| corner-reflector measurement and the session angle correction | `reflector.py` |
| orchestration, result object | `pipeline.py` |
| GUI, 3D replay, session tab | `gui.py`, `replay3d.py`, `session_tab.py` |

The headline metrics come from the joint range + angle fit under the chosen Tx slot shift. All three slot shifts stay viewable. For golf,
two more tiles give the club-head speed and smash factor (the club tile notes when the steepest
slope falls in a gap without edges, and the attack angle is assumed 0 ± 5°), and the other gives
the carry band over a interquartile range, labelled projected. Spin dependent metrics are for demonstration purposes only.
## 3D replay and projection

- **Green**: the fitted drag + gravity path from the tee to the last radar frame. It is
  constrained by the measurement.
- **Red**: the projection from that frame's position and velocity to landing, adding lift from an
  **assumed** spin. It is not a measurement.
- Carry, apex, offline distance and flight time are therefore projections, and the replay labels
  each value as measured, assumed or projected.
- Every assumed number lives in one table, `projection.ASSUMED`. Golf aerodynamics follow
  Smits & Smith (1994). The football and rugby lift model is nominal and untested.
- Playback: "Auto" runs the measured part at 0.1× and eases to 0.6×. Camera presets are behind
  the golfer, follow, side on, down the line and top down. **Export GIF** writes the replay with
  its overlays.

## Not in yet

- Measured spin 
- Spin axis
- Live processing while recording.

## Created By:
Chris Scheepers - under the supervision of Dr. Stephen Paine at the University of Cape Town

## Last Updated
25 October 2026
