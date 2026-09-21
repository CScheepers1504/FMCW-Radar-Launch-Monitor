# Sports-Ball-Analysis

Analysis FMCW radar ball launch monitor (EEE4022S, UCT, 2026).
Capture, driver and radar configuration live in
[Sports-Ball-Tracker](https://github.com/CScheepers1504/Sports-Ball-Tracker).

## State of Repo

Data capture and viability of accuracy has been determined.

Throwaway scripts written in MATLAB are being uploaded then rewritten and integrated with each other in Python with the view of processing on the Nvidia Jetson.

MATLAB files are officially unsupported from the 21st of September 2026.

## Input

Captures recorded by `recorder.py` in Sports-Ball-Tracker: `frames.bin`, `meta.json`, `index.csv`
per capture directory.
