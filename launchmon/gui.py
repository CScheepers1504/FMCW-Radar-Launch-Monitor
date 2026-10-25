"""Desktop interface: open captures, process them and inspect every stage.

python -m launchmon
"""

import csv
import json
import os
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from . import process
from .capture import is_capture
from .config import BALLS, PARAMS, Session, guess_ball
from .replay3d import Replay3D
from .session_tab import SessionTab

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
KMH = 3.6
SHIFT_PEN = {0: (66, 135, 245), 1: (245, 160, 66), 2: (90, 190, 110)}


def norm(path):
    return os.path.normpath(os.path.abspath(path))


class Worker(QtCore.QObject):
    """Processes a list of (capture, ball, window) jobs on a background thread."""

    progress = QtCore.pyqtSignal(str, float)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, jobs, session):
        super().__init__()
        self.jobs, self.session = jobs, session

    @QtCore.pyqtSlot()
    def run(self):
        for cap, ball, window in self.jobs:
            try:
                result = process(
                    cap,
                    self.session,
                    ball,
                    window=window,
                    progress=lambda stage, fraction: self.progress.emit(stage, fraction),
                )
                self.done.emit(result)
            except Exception:
                self.failed.emit(cap, traceback.format_exc())


class Tile(QtWidgets.QFrame):
    """One headline metric: title, value and a line of uncertainty or notes."""

    def __init__(self, title):
        super().__init__()
        self.setObjectName("tile")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        self.title = QtWidgets.QLabel(title.upper())
        self.title.setObjectName("tileTitle")
        self.value = QtWidgets.QLabel("–")
        self.value.setObjectName("tileValue")
        self.sub = QtWidgets.QLabel("")
        self.sub.setObjectName("tileSub")
        self.sub.setWordWrap(True)
        for label in (self.title, self.value, self.sub):
            layout.addWidget(label)

    def set(self, value, sub="", warn=False, rejected=False):
        self.value.setText(value)
        self.sub.setText(("REJECTED · " + sub) if rejected and value not in ("–", "") else sub)
        self.setProperty("warn", warn and not rejected)
        self.setProperty("rejected", rejected)
        # child labels are not re-polished with the parent
        self.value.setStyleSheet("color: #5d6573;" if rejected else "")
        self.style().unpolish(self)
        self.style().polish(self)


STYLE = """
QMainWindow, QWidget { background: #15181d; color: #dfe3ea; font-size: 10pt; }
QFrame#tile { background: #1f242c; border: 1px solid #2c333d; border-radius: 8px; }
QFrame#tile[warn="true"] { border: 1px solid #c98a1b; }
QFrame#tile QLabel { background: transparent; }
QFrame#tile[rejected="true"] QLabel#tileValue { color: #5d6573; }
QFrame#tile[rejected="true"] { border: 1px dashed #4a1d1d; }
QLabel#tileTitle { color: #8a94a6; font-size: 8pt; letter-spacing: 1px; }
QLabel#tileValue { font-size: 20pt; font-weight: 600; }
QLabel#tileSub { color: #9aa3b2; font-size: 8pt; }
QLabel#banner { padding: 6px 10px; border-radius: 6px; font-weight: 600; }
QTabBar::tab { background: #1f242c; padding: 6px 14px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #2b323c; }
QPushButton { background: #2b323c; border: 1px solid #3a434f; padding: 5px 12px; border-radius: 5px; }
QPushButton:hover { background: #34404f; }
QPushButton#primary { background: #2f6fdf; border: none; }
QComboBox, QPlainTextEdit { background: #1f242c; border: 1px solid #3a434f; padding: 3px; }
QProgressBar { background: #1f242c; border: 1px solid #3a434f; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #2f6fdf; }
"""


class Main(QtWidgets.QMainWindow):
    # share of the progress bar each processing stage fills
    STAGES = {
        "prepare": (0, 0.55),
        "detect": (0.55, 0.6),
        "range": (0.6, 0.65),
        "angles": (0.65, 1.0),
    }

    def __init__(self, session_path):
        super().__init__()
        self.setWindowTitle("Radar launch monitor")
        self.resize(1500, 920)
        self.session = Session.load(session_path)
        self.results = {}  # capture path -> ShotResult
        self.balls = {}  # capture path -> ball name
        self.current = None
        self.thread = None
        self._build()
        self._set_session_label()

    # Layout
    def _build(self):
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_folder = QtWidgets.QPushButton("Open folder…")
        open_capture = QtWidgets.QPushButton("Open capture…")
        open_session = QtWidgets.QPushButton("Session…")
        self.cap_box = QtWidgets.QComboBox()
        self.cap_box.setMinimumWidth(320)
        self.cap_box.currentIndexChanged.connect(lambda *_: self.on_select())
        self.ball = QtWidgets.QComboBox()
        self.ball.addItems(list(BALLS))
        self.b_run = QtWidgets.QPushButton("Process")
        self.b_run.setObjectName("primary")
        self.b_all = QtWidgets.QPushButton("Process all")
        export = QtWidgets.QPushButton("Export…")
        self.sess_lbl = QtWidgets.QLabel()
        for widget in (
            open_folder,
            open_capture,
            open_session,
            QtWidgets.QLabel("  Capture "),
            self.cap_box,
            QtWidgets.QLabel("  Ball "),
            self.ball,
            self.b_run,
            self.b_all,
            export,
        ):
            toolbar.addWidget(widget)
        toolbar.addSeparator()
        toolbar.addWidget(self.sess_lbl)
        open_folder.clicked.connect(self.open_folder)
        open_capture.clicked.connect(self.open_capture)
        open_session.clicked.connect(self.open_session)
        self.b_run.clicked.connect(self.run_selected)
        self.b_all.clicked.connect(self.run_all)
        export.clicked.connect(self.export)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        column = QtWidgets.QVBoxLayout(central)
        grid = QtWidgets.QGridLayout()
        # two rows of four, in the order commercial monitors use: club and ball, then where it went
        self.tiles = {
            key: Tile(title)
            for key, title in (
                ("club", "Club speed"),
                ("speed", "Ball speed"),
                ("smash", "Smash factor"),
                ("el", "Launch angle"),
                ("az", "Launch direction"),
                ("z", "Height at net"),
                ("y", "Lateral at net"),
                ("carry", "Carry (projected)"),
            )
        }
        for i, tile in enumerate(self.tiles.values()):
            grid.addWidget(tile, i // 4, i % 4)
        for col in range(4):
            grid.setColumnStretch(col, 1)
        column.addLayout(grid)
        self.banner = QtWidgets.QLabel("Open a folder of captures to begin.")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        column.addWidget(self.banner)

        self.tabs = QtWidgets.QTabWidget()
        column.addWidget(self.tabs, 1)
        self._build_tabs()

        bottom = QtWidgets.QHBoxLayout()
        self.prog = QtWidgets.QProgressBar()
        self.prog.setRange(0, 1000)
        self.stage = QtWidgets.QLabel("")
        bottom.addWidget(self.stage)
        bottom.addWidget(self.prog, 1)
        column.addLayout(bottom)

    def _plot(self, title, x_label, y_label):
        plot = pg.PlotWidget(title=title)
        plot.setBackground("#15181d")
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.addLegend(offset=(10, 10))
        return plot

    def _build_tabs(self):
        # range-time map
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.map_kind = QtWidgets.QComboBox()
        self.map_kind.addItems(["detection map", "clutter removed", "raw SNR", "raw power"])
        self.map_kind.currentIndexChanged.connect(lambda *_: self.show_map())
        self.b_window = QtWidgets.QPushButton("Re-run with window")
        self.b_window.setToolTip("Drag the shaded window on the map to override detection")
        self.b_window.clicked.connect(self.run_manual_window)
        self.show_trk = QtWidgets.QCheckBox("Track")
        self.show_trk.setChecked(True)
        self.show_trk.toggled.connect(lambda on: self.trk.setVisible(on))
        self.show_club = QtWidgets.QCheckBox("Club")
        self.show_club.setChecked(True)
        self.show_club.toggled.connect(
            lambda on: (self.club_pts.setVisible(on), self.club_fit.setVisible(on))
        )
        row.addWidget(self.map_kind)
        row.addWidget(self.show_trk)
        row.addWidget(self.show_club)
        row.addStretch(1)
        row.addWidget(self.b_window)
        layout.addLayout(row)
        self.p_map = self._plot("", "time in capture (s)", "range (m)")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.p_map.addItem(self.img)
        self.trk = pg.ScatterPlotItem(size=6, pen=pg.mkPen("r"), brush=pg.mkBrush(255, 60, 60, 160))
        self.p_map.addItem(self.trk)
        # club head, golf only: the edge picks and the fitted curve
        self.club_pts = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen((240, 240, 240)), brush=None, symbol="s"
        )
        self.club_fit = pg.PlotDataItem(pen=pg.mkPen((255, 170, 60), width=2))
        self.p_map.addItem(self.club_pts)
        self.p_map.addItem(self.club_fit)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(255, 255, 255, 25))
        self.p_map.addItem(self.region)
        layout.addWidget(self.p_map)
        self.tabs.addTab(page, "Range–time")

        # range fit
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.p_r = self._plot("range history", "time (s)", "range (m)")
        self.p_res = self._plot("residual", "time (s)", "mm")
        self.p_res.setMaximumHeight(220)
        layout.addWidget(self.p_r, 3)
        layout.addWidget(self.p_res, 1)
        self.tabs.addTab(page, "Range fit")

        # angles
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Tx slot shift"))
        self.shift_box = QtWidgets.QComboBox()
        self.shift_box.addItems(["chosen", "0", "1", "2"])
        self.shift_box.currentIndexChanged.connect(lambda *_: self.show_angles())
        row.addWidget(self.shift_box)
        row.addStretch(1)
        layout.addLayout(row)
        plots = QtWidgets.QHBoxLayout()
        self.p_el = self._plot("elevation", "range (m)", "deg")
        self.p_az = self._plot("azimuth (+ left)", "range (m)", "deg")
        plots.addWidget(self.p_el)
        plots.addWidget(self.p_az)
        layout.addLayout(plots)
        self.tabs.addTab(page, "Angles")

        # trajectory
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.p_side = self._plot("side view", "downrange (m)", "height (m)")
        self.p_top = self._plot("top view", "downrange (m)", "lateral (m, + left)")
        self.p_side.setAspectLocked(True)
        self.p_top.setAspectLocked(True)
        layout.addWidget(self.p_side)
        layout.addWidget(self.p_top)
        self.tabs.addTab(page, "Trajectory")

        self.replay = Replay3D()
        self.tabs.addTab(self.replay, "3D replay")
        self.tabs.currentChanged.connect(
            lambda i: self.tabs.widget(i) is self.replay and self.replay.replay()
        )

        self.session_tab = SessionTab(
            self.session, lambda: [self.cap_box.itemData(i) for i in range(self.cap_box.count())]
        )
        self.session_tab.applied.connect(self.set_session)
        self.tabs.addTab(self.session_tab, "Session")

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(QtGui.QFont("Consolas", 9))
        self.tabs.addTab(self.details, "Details")

    # Session and files
    def _set_session_label(self):
        s = self.session
        self.sess_lbl.setText(
            "  Session: %s  (radar %.2f m, tilt %+.1f°, tee %.2f m, net %.2f m)"
            % (
                s.name or os.path.basename(s.path),
                s.radar_height_m,
                s.tilt_deg,
                s.tee_xy_m[0],
                s.net_x_m,
            )
        )

    def open_session(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Session file", os.path.dirname(self.session.path), "JSON (*.json)"
        )
        if path:
            session = Session.load(path)
            self.session_tab.load_session(session)
            self.set_session(session)

    def set_session(self, session):
        """Use a new session. Earlier results used the old one, so they are dropped."""
        self.session = session
        self._set_session_label()
        self.results = {}
        for i in range(self.cap_box.count()):
            self.cap_box.setItemText(i, os.path.basename(self.cap_box.itemData(i)))
        self.clear_result("Session changed: press Process.")

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of captures")
        if folder:
            caps = sorted(
                os.path.join(folder, c)
                for c in os.listdir(folder)
                if is_capture(os.path.join(folder, c))
            )
            if is_capture(folder):
                caps = [folder]
            self.set_captures(caps)

    def open_capture(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Capture folder (frames.bin, index.csv, meta.json)"
        )
        if folder:
            if not is_capture(folder):
                QtWidgets.QMessageBox.warning(
                    self, "Not a capture", "Needs frames.bin, index.csv and meta.json."
                )
                return
            self.set_captures([folder])

    def set_captures(self, caps):
        # The file dialog returns "C:/..." but results are keyed by os.path.abspath ("C:\..."),
        # so both are normalised to one spelling.
        caps = [norm(c) for c in caps]
        self.cap_box.blockSignals(True)
        self.cap_box.clear()
        for cap in caps:
            self.balls.setdefault(cap, guess_ball(cap))
            self.cap_box.addItem(os.path.basename(cap), cap)
            self.cap_box.setItemData(self.cap_box.count() - 1, cap, Qt.ItemDataRole.ToolTipRole)
            if cap in self.results:
                self._fill_row(self.cap_box.count() - 1, self.results[cap])
        self.cap_box.blockSignals(False)
        if caps:
            self.cap_box.setCurrentIndex(0)
            self.on_select()

    def _row_of(self, cap):
        i = self.cap_box.findData(norm(cap))
        return None if i < 0 else i

    def selected_cap(self):
        return self.cap_box.currentData()

    def on_select(self):
        cap = self.selected_cap()
        if cap is None:
            return
        self.ball.setCurrentText(self.balls[cap])
        if cap in self.results:
            self.show_result(self.results[cap])
        else:
            self.clear_result("Not processed yet: press Process (or double-click).")

    # Processing
    def _start(self, jobs):
        if self.thread is not None:
            return
        self.thread = QtCore.QThread()
        self.worker = Worker(jobs, self.session)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self._remaining = len(jobs)
        self.b_run.setEnabled(False)
        self.b_all.setEnabled(False)
        self.thread.start()

    def _finish_one(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.b_run.setEnabled(True)
            self.b_all.setEnabled(True)
            self.stage.setText("done")
            self.prog.setValue(1000)

    def run_selected(self):
        cap = self.selected_cap()
        if cap:
            self.balls[cap] = self.ball.currentText()
            self._start([(cap, self.ball.currentText(), None)])

    def run_all(self):
        caps = [self.cap_box.itemData(i) for i in range(self.cap_box.count())]
        jobs = [(cap, self.balls[cap], None) for cap in caps]
        if jobs:
            self._start(jobs)

    def run_manual_window(self):
        cap = self.selected_cap()
        if cap:
            lo, hi = self.region.getRegion()
            self._start([(cap, self.ball.currentText(), (float(lo), float(hi)))])

    def on_progress(self, stage, fraction):
        lo, hi = self.STAGES.get(stage, (0, 1))
        self.prog.setValue(int(1000 * (lo + (hi - lo) * fraction)))
        self.stage.setText(stage)

    def on_done(self, result):
        cap = norm(result.capture)
        self.results[cap] = result
        row = self._row_of(cap)
        if row is not None:
            self._fill_row(row, result)
        if row is None or self.selected_cap() == cap:
            self.show_result(result)
        self._finish_one()

    def on_failed(self, cap, trace):
        self.details.setPlainText(trace)
        row = self._row_of(cap)
        if row is not None:
            self.cap_box.setItemText(row, "%s  ·  failed" % os.path.basename(cap))
        self.banner.setText("Processing failed — traceback in Details.")
        self._banner_style("fail")
        self._finish_one()

    def _fill_row(self, row, result):
        speed = "" if result.speed is None else "  ·  %.0f km/h" % (result.speed * KMH)
        self.cap_box.setItemText(
            row, "%s  ·  %s%s" % (os.path.basename(result.capture), result.status.lower(), speed)
        )

    # Display
    def _banner_style(self, kind):
        colour = {"ok": "#1d3b2a", "warn": "#4a3a17", "fail": "#4a1d1d", "idle": "#1f242c"}[kind]
        self.banner.setStyleSheet("background: %s;" % colour)

    def clear_result(self, message):
        self.current = None
        for tile in self.tiles.values():
            tile.set("–")
        self.banner.setText(message)
        self._banner_style("idle")
        for plot in (self.p_r, self.p_res, self.p_el, self.p_az, self.p_side, self.p_top):
            plot.clear()
        self.img.clear()
        self.trk.clear()
        self.club_pts.clear()
        self.club_fit.clear()
        self.replay.set_result(None, self.session)
        self.details.clear()

    def show_result(self, result):
        self.current = result
        tiles = self.tiles
        if result.speed is not None:
            tiles["speed"].set(
                "%.1f m/s" % result.speed,
                "%.0f km/h · ±%.2f m/s (%.2f %%)"
                % (result.speed * KMH, result.speed_sigma, 100 * result.speed_sigma / result.speed),
            )
            elevation_doubtful = not result.launch_el_measured or not result.slot_decided
            if result.launch_el_measured:
                el_note = "±%.1f° fit · ±%.1f° sys" % (
                    result.launch_el_sigma,
                    result.launch_el_sys or np.nan,
                )
            else:
                el_note = "NOT MEASURED — from range only"
            if not result.slot_decided:
                el_note += " · Tx order undecided"
            tiles["el"].set("%.1f°" % result.launch_el, el_note, warn=elevation_doubtful)
            side = "L" if result.launch_az > 0 else "R"
            tiles["az"].set(
                "%.1f° %s" % (abs(result.launch_az), side),
                "±%.1f° fit · ±%.1f° sys"
                % (result.launch_az_sigma, result.launch_az_sys or np.nan),
            )
            net = result.net
            if net.get("z") is not None:
                tiles["z"].set(
                    "%.2f m" % net["z"],
                    "±%.2f fit · ±%.2f sys" % (net["z_formal"] or np.nan, net["z_sys"] or np.nan),
                    warn=elevation_doubtful,
                )
                tiles["y"].set(
                    "%.2f m %s" % (abs(net["y"]), "L" if net["y"] > 0 else "R"),
                    "±%.2f fit · ±%.2f sys" % (net["y_formal"] or np.nan, net["y_sys"] or np.nan),
                )
            else:
                tiles["z"].set("–", "path does not reach the net")
                tiles["y"].set("–")
            self._golf_tiles(result)
        else:
            for tile in tiles.values():
                tile.set("–")
            if result.speed_range_only is not None:
                tiles["speed"].set("%.1f m/s" % result.speed_range_only, "range only", warn=True)
        if result.status != "ACCEPT":  # keep the numbers for diagnosis, but never as a measurement
            for tile in tiles.values():
                tile.set(tile.value.text(), tile.sub.text(), rejected=True)
        message = "%s — %s" % (
            result.status,
            ("; ".join(result.reasons) if result.reasons else "all acceptance tests passed"),
        )
        if result.detection.get("manual"):
            message += "  [MANUAL WINDOW %.2f–%.2f s]" % tuple(result.detection["window"])
        self.banner.setText(message)
        if result.status == "ACCEPT" and not result.reasons:
            self._banner_style("ok")
        else:
            self._banner_style("warn" if result.status in ("ACCEPT", "REJECT") else "fail")
        self.show_map()
        self.show_range()
        self.show_angles()
        self.show_traj()
        self.replay.set_result(result, self.session)
        self.details.setPlainText(self._details_text(result))
        if self.tabs.currentWidget() is self.replay:
            self.replay.replay()

    def _golf_tiles(self, result):
        """Club speed, smash factor and carry tiles, golf only."""
        tiles = self.tiles
        if result.ball != "golf":
            for key in ("club", "smash", "carry"):
                tiles[key].set("–", "golf only")
            return
        if result.club_speed is not None:
            club = result.club
            gap_note = (
                "" if club["edges_at_slope"] else " · slope in a %.0f ms gap" % club["gap_ms"]
            )
            tiles["club"].set(
                "%.1f m/s" % result.club_speed,
                "%.0f km/h · ±%.1f m/s%s"
                % (result.club_speed * KMH, result.club_speed_sigma, gap_note),
            )
            tiles["club"].setToolTip(
                "radial %.1f m/s at %+.1f ms, line of sight %.0f° down, attack angle 0 ± %g° assumed\n"
                "%d edges · resampled ±%.1f %% · 6 dB edge %+.1f %%"
                % (
                    club["radial"],
                    club["t_peak_ms"],
                    club["los_depression_deg"],
                    PARAMS["club_alpha_deg"],
                    club["n_edges"],
                    club["edge_sd_pct"],
                    club["thr_change_pct"] or 0.0,
                )
            )
            tiles["smash"].set(
                "%.2f" % result.smash, "±%.2f · ball / club speed" % result.smash_sigma
            )
        else:
            tiles["club"].set("–", "club head not found")
            tiles["smash"].set("–")
        if result.carry:
            band = sorted(result.carry.items())
            tiles["carry"].set(
                "%.0f–%.0f m" % (band[0][1], band[-1][1]),
                "spin %d–%d rpm assumed · not measured%s"
                % (
                    band[0][0],
                    band[-1][0],
                    "" if result.launch_el_measured else " · elevation from range only",
                ),
                warn=not result.launch_el_measured,
            )
        else:
            tiles["carry"].set("–")

    def show_map(self):
        result = self.current
        if result is None or "snr" not in result._arrays:
            return
        arrays = result._arrays
        kind = self.map_kind.currentIndex()
        image = {
            0: arrays["dmap"],
            1: arrays["snr"],
            2: arrays.get("raw"),
            3: arrays.get("raw_pw"),
        }[kind]
        if image is None:  # result from before the raw maps existed
            self.img.clear()
            return
        t, rbin = arrays["t"], arrays["rbin"]
        window = result.detection["window"]
        centre = 0.5 * (window[0] + window[1])
        first, last = np.searchsorted(t, centre - 1.0), np.searchsorted(t, centre + 1.0)
        shown = np.nan_to_num(image[first:last])
        # the raw maps span tens of dB of static clutter, so they get their own scale
        levels = {
            2: (0.0, float(np.percentile(shown, 99.5))),
            3: (float(np.percentile(shown, 5)), float(np.percentile(shown, 99.5))),
        }.get(kind, (0.0, 25.0))
        shown = np.clip(shown, *levels).T
        self.img.setImage(shown, levels=levels)
        self.img.setRect(
            QtCore.QRectF(
                float(t[first]), 0.0, float(t[last - 1] - t[first]), shown.shape[0] * rbin
            )
        )
        self.region.setRegion(window)
        track = arrays.get("track")
        if track is not None and len(track["fr"]):
            self.trk.setData(t[track["fr"]], track["rr"])
        else:
            self.trk.clear()
        club = arrays.get("club")
        if club is not None and result.t_launch is not None:
            from .club import gompertz

            t_launch = result.shifts[result.slot_shift]["x"][3]
            edges = np.asarray(club["edges"])
            self.club_pts.setData(edges[:, 0] + t_launch, edges[:, 1])
            tg = np.linspace(edges[:, 0].min(), edges[:, 0].max(), 300)
            self.club_fit.setData(tg + t_launch, gompertz(tg, *club["curve"]))
        else:
            self.club_pts.clear()
            self.club_fit.clear()
        self.p_map.setYRange(0, 7)
        self.p_map.setTitle("")

    def show_range(self):
        result = self.current
        self.p_r.clear()
        self.p_res.clear()
        if result is None or "track" not in result._arrays:
            return
        track, t = result._arrays["track"], result._arrays["t"]
        tf = t[track["fr"]]
        self.p_r.plot(
            tf, track["rr"], pen=None, symbol="o", symbolSize=5, symbolBrush="w", name="measured"
        )
        curves = result._arrays.get("curves")
        if curves:
            shift = result.slot_shift
            fit = curves[shift]
            self.p_r.plot(
                fit["t"],
                fit["r"],
                pen=pg.mkPen(SHIFT_PEN[shift], width=2),
                name="joint fit (shift %d)" % shift,
            )
            self.p_r.plot(
                curves["range_only"]["t"],
                curves["range_only"]["r"],
                pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DashLine),
                name="range-only fit",
            )
            residual_mm = 1e3 * (track["rr"] - np.interp(tf, fit["t"], fit["r"]))
            self.p_res.plot(
                tf, residual_mm, pen=None, symbol="o", symbolSize=5, symbolBrush=SHIFT_PEN[shift]
            )
            self.p_res.addLine(y=0, pen=pg.mkPen((120, 120, 120)))
        row = result.range_row
        self.p_r.setTitle(
            "range-only: %s m/s ±%s %%, rms %s mm, n %s, span %s m"
            % (
                row.get("v_geo", "–"),
                row.get("sv_pct", "–"),
                row.get("rms_mm", "–"),
                row.get("n", "–"),
                row.get("span", "–"),
            )
        )

    def show_angles(self):
        result = self.current
        self.p_el.clear()
        self.p_az.clear()
        if result is None or "frames" not in result._arrays:
            return
        choice = self.shift_box.currentIndex()
        shift = result.slot_shift if choice == 0 else choice - 1
        frames = result._arrays["frames_by_shift"][shift]
        ranges = result._arrays["frames"]["r"]
        lo, hi = self.session.el_valid_span_deg
        self.p_el.addItem(
            pg.LinearRegionItem(
                (lo, hi),
                orientation="horizontal",
                movable=False,
                brush=pg.mkBrush(90, 190, 110, 30),
            )
        )
        used = frames["use_el"]
        self.p_el.plot(
            ranges[used],
            frames["el"][used],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush="w",
            name="used",
        )
        self.p_el.plot(
            ranges[~used],
            frames["el"][~used],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=None,
            symbolPen=(150, 150, 150),
            name="outside calibrated span / outlier",
        )
        self.p_az.plot(ranges, frames["az"], pen=None, symbol="o", symbolSize=6, symbolBrush="w")
        fit = result._arrays["curves"][shift]
        self.p_el.plot(
            fit["r"], fit["el"], pen=pg.mkPen(SHIFT_PEN[shift], width=2), name="joint fit"
        )
        self.p_az.plot(fit["r"], fit["az"], pen=pg.mkPen(SHIFT_PEN[shift], width=2))
        fitted = result.shifts[shift]
        self.p_el.setTitle(
            "shift %d%s: el %.1f ± %.1f°, %d frames, slot off %+.0f° R %.2f"
            % (
                shift,
                " (chosen)" if shift == result.slot_shift else "",
                fitted["el"],
                fitted["sel"],
                fitted["n_el"],
                fitted["slot_offset_deg"],
                fitted["slot_R"],
            )
        )
        self.p_az.setTitle("launch az %+.1f ± %.1f°" % (fitted["az"], fitted["saz"]))

    def show_traj(self):
        result = self.current
        self.p_side.clear()
        self.p_top.clear()
        if result is None or "path" not in result._arrays:
            return
        session = self.session
        net_x = session.net_x_m
        for shift, (_, pos) in result._arrays["path"].items():
            before_net = (pos[:, 0] <= net_x + 0.05) & (pos[:, 2] > -0.05)
            chosen = shift == result.slot_shift
            pen = pg.mkPen(
                SHIFT_PEN[shift],
                width=2.5 if chosen else 1,
                style=Qt.PenStyle.SolidLine if chosen else Qt.PenStyle.DotLine,
            )
            name = "shift %d%s" % (shift, " (chosen)" if chosen else "")
            self.p_side.plot(pos[before_net, 0], pos[before_net, 2], pen=pen, name=name)
            self.p_top.plot(pos[before_net, 0], pos[before_net, 1], pen=pen)
        frames = result._arrays["frames"]
        xyz, used = frames["xyz"], frames["el_used"]
        self.p_side.addItem(
            pg.ErrorBarItem(
                x=xyz[used, 0],
                y=xyz[used, 2],
                height=2 * frames["sig_vert_m"][used],
                pen=(200, 200, 200),
            )
        )
        self.p_top.addItem(
            pg.ErrorBarItem(
                x=xyz[:, 0], y=xyz[:, 1], height=2 * frames["sig_lat_m"], pen=(200, 200, 200)
            )
        )
        self.p_side.plot(
            xyz[used, 0],
            xyz[used, 2],
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush="w",
            name="frames",
        )
        self.p_side.plot(
            xyz[~used, 0],
            xyz[~used, 2],
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=None,
            symbolPen=(150, 150, 150),
        )
        self.p_top.plot(xyz[:, 0], xyz[:, 1], pen=None, symbol="o", symbolSize=5, symbolBrush="w")
        for plot, (y0, y1) in ((self.p_side, (0, 2.5)), (self.p_top, (-1.5, 1.5))):
            plot.plot([net_x, net_x], [y0, y1], pen=pg.mkPen((220, 70, 70), width=2))
            plot.plot(
                [0],
                [session.radar_height_m if plot is self.p_side else 0],
                pen=None,
                symbol="s",
                symbolBrush=(220, 70, 70),
            )
        tee_height = session.ball_centre_height_m.get(
            result.ball, BALLS[result.ball].centre_height_default
        )
        self.p_side.plot(
            [session.tee_xy_m[0]], [tee_height], pen=None, symbol="t", symbolBrush=(90, 190, 110)
        )
        self.p_top.plot(
            [session.tee_xy_m[0]],
            [session.tee_xy_m[1]],
            pen=None,
            symbol="t",
            symbolBrush=(90, 190, 110),
        )
        if result.net.get("z") is not None:
            self.p_side.plot(
                [net_x],
                [result.net["z"]],
                pen=None,
                symbol="x",
                symbolSize=14,
                symbolPen=pg.mkPen("y", width=2),
            )
            self.p_top.plot(
                [net_x],
                [result.net["y"]],
                pen=None,
                symbol="x",
                symbolSize=14,
                symbolPen=pg.mkPen("y", width=2),
            )
        self.p_side.setXRange(-0.2, net_x + 0.4)
        self.p_side.setYRange(0, 2.5)
        self.p_top.setXRange(-0.2, net_x + 0.4)
        self.p_top.setYRange(-1.5, 1.5)

    def _details_text(self, result):
        lines = [result.summary(), ""]
        if result.shifts:
            lines.append("per Tx slot shift (joint fit):")
            lines.append(
                "  shift   v m/s    el deg      az deg     n_el  slot off  R     net z   net y"
            )
            for shift, fitted in sorted(result.shifts.items()):
                net = fitted["net"]
                lines.append(
                    "  %d%s     %6.2f  %5.1f±%-4.1f  %+6.1f±%-4.1f  %3d   %+6.0f  %.2f  %6s  %6s"
                    % (
                        shift,
                        "*" if shift == result.slot_shift else " ",
                        fitted["v"],
                        fitted["el"],
                        fitted["sel"],
                        fitted["az"],
                        fitted["saz"],
                        fitted["n_el"],
                        fitted["slot_offset_deg"],
                        fitted["slot_R"],
                        "%.2f" % net["z"] if net["z"] is not None else "–",
                        "%+.2f" % net["y"] if net["y"] is not None else "–",
                    )
                )
            lines.append("")
        lines.append("range stage: " + json.dumps(result.range_row))
        lines.append("detection:   " + json.dumps(result.detection))
        lines.append("capture:     " + json.dumps(result.prep_info))
        lines.append(
            "timings s:   " + json.dumps({k: round(v, 2) for k, v in result.timings.items()})
        )
        lines.append("session:     " + self.session.path)
        return "\n".join(lines)

    def export(self):
        if not self.results:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export results to")
        if not folder:
            return
        rows = []
        for cap, result in self.results.items():
            result.save_json(os.path.join(folder, os.path.basename(cap) + ".launch.json"))
            rows.append(
                dict(
                    capture=os.path.basename(cap),
                    ball=result.ball,
                    status=result.status,
                    reasons="; ".join(result.reasons),
                    speed=result.speed,
                    speed_sigma=result.speed_sigma,
                    launch_el=result.launch_el,
                    launch_el_sigma=result.launch_el_sigma,
                    launch_el_sys=result.launch_el_sys,
                    el_measured=result.launch_el_measured,
                    launch_az=result.launch_az,
                    launch_az_sigma=result.launch_az_sigma,
                    launch_az_sys=result.launch_az_sys,
                    net_z=result.net.get("z"),
                    net_y=result.net.get("y"),
                    slot_shift=result.slot_shift,
                    slot_decided=result.slot_decided,
                    manual_window=bool(result.detection.get("manual")),
                    club_speed=result.club_speed,
                    club_speed_sigma=result.club_speed_sigma,
                    club_radial=result.club.get("radial"),
                    club_edges_at_slope=result.club.get("edges_at_slope"),
                    smash=result.smash,
                    smash_sigma=result.smash_sigma,
                    **{"carry_m_%drpm" % rpm: carry for rpm, carry in sorted(result.carry.items())},
                )
            )
        with open(os.path.join(folder, "launch_results.csv"), "w", newline="") as f:
            # golf rows add club and carry columns
            columns = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        self.stage.setText("exported %d results to %s" % (len(rows), folder))


def run(session_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    window = Main(session_path)
    window.show()
    return app.exec()
