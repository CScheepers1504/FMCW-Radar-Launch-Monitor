"""Session tab: edit the rig survey, measure reflectors, set the angle correction and save."""

import json
import os
import traceback

from PyQt6 import QtCore, QtGui, QtWidgets

from . import reflector as RF
from .config import BALLS, Session

OK, BAD, WARN = "#5cc97a", "#e05a5a", "#e0a43a"
RCOLS = [
    "x m",
    "y m",
    "z m",
    "capture",
    "found",
    "SNR dB",
    "range err m",
    "az err °",
    "el err °",
    "slot order",
    "use",
]


def _spin(lo, hi, dec, step, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    # "0.520", matching the JSON and the table, whatever the OS locale
    s.setLocale(QtCore.QLocale.c())
    return s


class FnWorker(QtCore.QObject):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


class SessionTab(QtWidgets.QWidget):
    applied = QtCore.pyqtSignal(object)  # the Session processing should now use

    def __init__(self, session, captures_fn):
        super().__init__()
        self.captures_fn = captures_fn  # -> paths of the captures loaded in the main window
        self.results = []  # per reflector row: reflector.measure() result or None
        self.thread = None
        self.path = ""
        self._build()
        self.load_session(session)

    # Layout
    def _build(self):
        top = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        g = QtWidgets.QGroupBox("Survey")
        f = QtWidgets.QFormLayout(g)
        self.name = QtWidgets.QLineEdit()
        self.height = _spin(0, 5, 3, 0.01, " m")
        self.tilt = _spin(-45, 45, 2, 0.5, "°")
        self.yaw = _spin(-45, 45, 2, 0.5, "°")
        self.tee_x = _spin(0, 20, 3, 0.01, " m")
        self.tee_y = _spin(-5, 5, 3, 0.01, " m")
        self.net_x = _spin(0, 50, 2, 0.05, " m")
        self.tilt.setToolTip("boresight pitch, + up")
        self.yaw.setToolTip("boresight yaw, + left of the target line")
        self.tee_x.setToolTip("downrange from the antenna face")
        self.tee_y.setToolTip("lateral, + left looking downrange")
        for lab, w in (
            ("Name", self.name),
            ("Radar height", self.height),
            ("Tilt", self.tilt),
            ("Yaw", self.yaw),
            ("Tee downrange", self.tee_x),
            ("Tee lateral", self.tee_y),
            ("Net downrange", self.net_x),
        ):
            f.addRow(lab, w)
        self.ball_h = {}
        for b, spec in BALLS.items():
            self.ball_h[b] = _spin(0, 1, 3, 0.005, " m")
            self.ball_h[b].setToolTip("ball centre above the floor (tee height + radius)")
            f.addRow("%s centre height" % b.capitalize(), self.ball_h[b])
        left.addWidget(g)

        g = QtWidgets.QGroupBox("Calibration")
        f = QtWidgets.QFormLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.cal = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(self._browse_cal)
        row.addWidget(self.cal, 1)
        row.addWidget(b)
        f.addRow("Array calibration", row)
        self.el_lo, self.el_hi = _spin(-30, 30, 1, 0.1, "°"), _spin(-30, 30, 1, 0.1, "°")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.el_lo)
        row.addWidget(QtWidgets.QLabel("to"))
        row.addWidget(self.el_hi)
        f.addRow("Validated elevation", row)
        left.addWidget(g)

        g = QtWidgets.QGroupBox("Notes")
        v = QtWidgets.QVBoxLayout(g)
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(90)
        v.addWidget(self.notes)
        left.addWidget(g)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        g = QtWidgets.QGroupBox("Corner reflectors")
        v = QtWidgets.QVBoxLayout(g)
        self.tbl = QtWidgets.QTableWidget(0, len(RCOLS))
        self.tbl.setHorizontalHeaderLabels(RCOLS)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.tbl.cellDoubleClicked.connect(self._cell_clicked)
        self.tbl.itemChanged.connect(lambda it: it.column() == 10 and self._update_proposal())
        v.addWidget(self.tbl)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (
            ("Add", self._add_row),
            ("Remove", self._remove_row),
            ("Measure", self.measure),
            ("Check loaded shots", self.check_shots),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        hint = QtWidgets.QLabel(
            "Apex position, world frame: x downrange, y left, z up from the floor. "
            "Double-click the capture cell to pick the reflector take."
        )
        hint.setObjectName("tileSub")
        hint.setWordWrap(True)
        v.addWidget(hint)
        right.addWidget(g)

        g = QtWidgets.QGroupBox(
            "Angle correction (measured − surveyed, subtracted from every frame)"
        )
        f = QtWidgets.QFormLayout(g)
        self.corr_az, self.corr_el = _spin(-20, 20, 2, 0.1, "°"), _spin(-20, 20, 2, 0.1, "°")
        self.sys = _spin(0, 10, 2, 0.1, "°")
        self.sys.setToolTip("bound on the correction; refits use ± this")
        self.corr_src = QtWidgets.QLineEdit()
        self.b_use = QtWidgets.QPushButton("Use measured correction")
        self.b_use.setEnabled(False)
        self.b_use.clicked.connect(self._use_measured)
        self.proposal = QtWidgets.QLabel("")
        self.proposal.setWordWrap(True)
        f.addRow("Azimuth", self.corr_az)
        f.addRow("Elevation", self.corr_el)
        f.addRow("Systematic", self.sys)
        f.addRow("Source", self.corr_src)
        f.addRow(self.b_use, self.proposal)
        right.addWidget(g)

        g = QtWidgets.QGroupBox("Reflector in the shots")
        v = QtWidgets.QVBoxLayout(g)
        self.shots = QtWidgets.QTableWidget(0, 1)
        self.shots.verticalHeader().setVisible(False)
        self.shots.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        v.addWidget(self.shots)
        right.addWidget(g, 1)

        row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        row.addWidget(self.status, 1)
        for text, fn, prim in (
            ("Load…", self._load, False),
            ("Save as…", self._save, False),
            ("Apply", self.apply, True),
        ):
            b = QtWidgets.QPushButton(text)
            if prim:
                b.setObjectName("primary")
            b.clicked.connect(fn)
            row.addWidget(b)
        right.addLayout(row)

        top.addLayout(left, 2)
        top.addLayout(right, 3)

    # Session <-> form
    def load_session(self, s):
        self.path = s.path
        self.name.setText(s.name)
        self.height.setValue(s.radar_height_m)
        self.tilt.setValue(s.tilt_deg)
        self.yaw.setValue(s.yaw_deg)
        self.tee_x.setValue(s.tee_xy_m[0])
        self.tee_y.setValue(s.tee_xy_m[1])
        self.net_x.setValue(s.net_x_m)
        for b, w in self.ball_h.items():
            w.setValue(s.ball_centre_height_m.get(b, BALLS[b].centre_height_default))
        self.cal.setText(s.calibration)
        self.el_lo.setValue(s.el_valid_span_deg[0])
        self.el_hi.setValue(s.el_valid_span_deg[1])
        self.notes.setPlainText(s.notes)
        c = s.angle_correction_deg
        self.corr_az.setValue(c.get("az", 0.0))
        self.corr_el.setValue(c.get("el", 0.0))
        self.corr_src.setText(c.get("source", ""))
        self.sys.setValue(s.angle_systematic_deg)
        self.tbl.setRowCount(0)
        self.results = []
        for r in s.reflectors:
            self._add_row(xyz=r.get("xyz", (0, 0, 0)), cap=r.get("capture", ""))
        self.shots.setRowCount(0)
        self._update_proposal()
        self.status.setText(os.path.basename(s.path) if s.path else "unsaved")

    def session(self):
        """A new Session from the form (the loaded one is never modified)."""
        s = Session(
            name=self.name.text(),
            radar_height_m=self.height.value(),
            tilt_deg=self.tilt.value(),
            yaw_deg=self.yaw.value(),
            tee_xy_m=(self.tee_x.value(), self.tee_y.value()),
            net_x_m=self.net_x.value(),
            calibration=self.cal.text(),
            el_valid_span_deg=(self.el_lo.value(), self.el_hi.value()),
            angle_correction_deg={
                "az": self.corr_az.value(),
                "el": self.corr_el.value(),
                "source": self.corr_src.text(),
            },
            angle_systematic_deg=self.sys.value(),
            ball_centre_height_m={b: w.value() for b, w in self.ball_h.items()},
            reflectors=self._reflectors(),
            notes=self.notes.toPlainText(),
        )
        s.path = self.path
        return s

    def _reflectors(self):
        out = []
        for i in range(self.tbl.rowCount()):
            try:
                xyz = [float(self.tbl.item(i, k).text()) for k in range(3)]
            except (AttributeError, ValueError):
                raise ValueError("reflector %d: x, y, z must be numbers" % (i + 1))
            out.append(
                {"xyz": xyz, "capture": (self.tbl.item(i, 3).text() if self.tbl.item(i, 3) else "")}
            )
        return out

    # Reflector table
    def _add_row(self, *_, xyz=(0.0, 0.0, 0.0), cap=""):
        i = self.tbl.rowCount()
        self.tbl.insertRow(i)
        for k, v in enumerate(xyz):
            self.tbl.setItem(i, k, QtWidgets.QTableWidgetItem("%.3f" % v))
        it = QtWidgets.QTableWidgetItem(os.path.basename(cap))
        it.setData(QtCore.Qt.ItemDataRole.UserRole, cap)
        it.setToolTip(cap)
        it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.tbl.setItem(i, 3, it)
        use = QtWidgets.QTableWidgetItem()
        use.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        use.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.tbl.setItem(i, 10, use)
        self.results.append(None)

    def _remove_row(self):
        i = self.tbl.currentRow()
        if i >= 0:
            self.tbl.removeRow(i)
            del self.results[i]
            self._update_proposal()

    def _cell_clicked(self, i, col):
        if col != 3:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Reflector capture")
        if d:
            it = self.tbl.item(i, 3)
            it.setText(os.path.basename(d))
            it.setData(QtCore.Qt.ItemDataRole.UserRole, os.path.normpath(d))
            it.setToolTip(d)

    def _cap_of(self, i):
        it = self.tbl.item(i, 3)
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else ""

    def _set(self, i, col, text, color=None):
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        if color:
            it.setForeground(QtGui.QColor(color))
        self.tbl.setItem(i, col, it)

    def _show_result(self, i, r):
        self.results[i] = r
        if r is None:
            for col in range(4, 10):
                self._set(i, col, "")
            return
        self._set(i, 4, "yes" if r["present"] else "no", OK if r["present"] else BAD)
        self._set(i, 5, "%.1f" % r["snr_db"])
        if r["present"]:  # an absent reflector's "errors" are just the noise peak's
            self._set(i, 6, "%+.3f" % r["range_err_m"])
            self._set(i, 7, "%+.2f" % r["az_err_deg"])
            self._set(i, 8, "%+.2f" % r["el_err_deg"])
            self._set(
                i,
                9,
                "decided" if r["verdict"] == "CONFIDENT" else "undecided",
                OK if r["verdict"] == "CONFIDENT" else WARN,
            )
        else:
            for col in range(6, 10):
                self._set(i, col, "–")
        use = self.tbl.item(i, 10)
        use.setCheckState(
            QtCore.Qt.CheckState.Checked if r["usable"] else QtCore.Qt.CheckState.Unchecked
        )

    # Background jobs
    def _run(self, label, fn, on_done):
        if self.thread is not None:
            return
        self.status.setText(label + "…")
        self.thread = QtCore.QThread()
        self.worker = FnWorker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        def finish(*_):
            self.thread.quit()
            self.thread.wait()
            self.thread = None

        self.worker.done.connect(lambda res: (finish(), on_done(res)))
        self.worker.failed.connect(lambda tb: (finish(), self._fail(tb)))
        self.thread.start()

    def _fail(self, tb):
        self.status.setText("failed: " + tb.strip().splitlines()[-1])
        QtWidgets.QMessageBox.warning(self, "Reflector analysis failed", tb[-2000:])

    def _inputs(self):
        try:
            s = self.session()
            with open(s.calibration) as f:
                cal = json.load(f)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Session", str(e))
            return None, None
        return s, cal

    def measure(self):
        s, cal = self._inputs()
        if s is None:
            return
        jobs = [
            (i, self._cap_of(i), r["xyz"]) for i, r in enumerate(s.reflectors) if self._cap_of(i)
        ]
        if not jobs:
            self.status.setText("pick a reflector capture first (double-click the capture cell)")
            return
        self._run(
            "measuring %d reflector(s)" % len(jobs),
            lambda: [(i, RF.measure(c, cal, xyz, s)) for i, c, xyz in jobs],
            self._measured,
        )

    def _measured(self, res):
        for i, r in res:
            if i < self.tbl.rowCount():
                self._show_result(i, r)
        self._update_proposal()
        n = sum(r["present"] for _, r in res)
        self.status.setText("%d of %d reflector(s) found" % (n, len(res)))

    def check_shots(self):
        s, cal = self._inputs()
        if s is None:
            return
        caps = self.captures_fn()
        if not caps or not s.reflectors:
            self.status.setText("needs loaded captures and at least one reflector")
            return
        xyzs = [r["xyz"] for r in s.reflectors]
        self._run(
            "checking %d shot(s)" % len(caps),
            lambda: [(c, [RF.measure(c, cal, xyz, s) for xyz in xyzs]) for c in caps],
            self._checked,
        )

    def _checked(self, res):
        n = len(res[0][1]) if res else 0
        self.shots.setColumnCount(1 + n)
        self.shots.setHorizontalHeaderLabels(
            ["capture"] + ["reflector %d" % (k + 1) for k in range(n)]
        )
        self.shots.setRowCount(len(res))
        missing = 0
        for i, (cap, rs) in enumerate(res):
            self.shots.setItem(i, 0, QtWidgets.QTableWidgetItem(os.path.basename(cap)))
            for k, r in enumerate(rs):
                it = QtWidgets.QTableWidgetItem(
                    "%s  %.0f dB" % ("yes" if r["present"] else "no", r["snr_db"])
                )
                it.setForeground(QtGui.QColor(OK if r["present"] else BAD))
                self.shots.setItem(i, 1 + k, it)
            missing += not any(r["present"] for r in rs)
        self.shots.resizeColumnsToContents()
        self.status.setText(
            "no reflector in %d of %d shot(s)" % (missing, len(res))
            if missing
            else "reflector present in every shot"
        )

    # Correction
    def _used(self):
        return [
            r
            for i, r in enumerate(self.results)
            if r is not None
            and self.tbl.item(i, 10) is not None
            and self.tbl.item(i, 10).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def _update_proposal(self):
        c = RF.correction(self._used())
        self.b_use.setEnabled(c is not None)
        if c is None:
            self.proposal.setText("measure a reflector to get one")
            return
        txt = "az %+.2f°, el %+.2f° from %d reflector(s)" % (c["az"], c["el"], c["n"])
        txt += (
            ", spread %.2f°" % c["spread"]
            if c["spread"] is not None
            else " — one reflector, so no spread to bound it"
        )
        self.proposal.setText(txt)

    def _use_measured(self):
        c = RF.correction(self._used())
        if c is None:
            return
        self.corr_az.setValue(c["az"])
        self.corr_el.setValue(c["el"])
        self.corr_src.setText(c["source"])
        if c["spread"] is not None:
            self.sys.setValue(max(c["spread"], 0.5))

    # Files
    def _browse_cal(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Array calibration", os.path.dirname(self.cal.text()), "JSON (*.json)"
        )
        if p:
            self.cal.setText(os.path.normpath(p))

    def _load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Session file", os.path.dirname(self.path), "JSON (*.json)"
        )
        if p:
            self.load_session(Session.load(p))

    def _save(self):
        try:
            s = self.session()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Session", str(e))
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", self.path or "session.json", "JSON (*.json)"
        )
        if p:
            s.save(p)
            self.path = os.path.abspath(p)
            self.status.setText("saved " + os.path.basename(p))

    def apply(self):
        try:
            s = self.session()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Session", str(e))
            return
        if not os.path.isfile(s.calibration):
            QtWidgets.QMessageBox.warning(
                self, "Session", "Calibration file not found:\n%s" % s.calibration
            )
            return
        self.applied.emit(s)
        self.status.setText("applied — reprocess the captures")
