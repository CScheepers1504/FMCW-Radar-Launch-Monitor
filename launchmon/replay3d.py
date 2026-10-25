"""3D replay of a shot: the measured path in green, the projection to landing in red."""

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui, QtWidgets

from .projection import ASSUMED, project

GREEN = (0.25, 0.95, 0.35, 1.0)
RED = (1.0, 0.25, 0.2, 1.0)
SPEEDS = ["Auto (slow near tee)", "0.1×", "0.25×", "0.5×", "1×"]
CAMERAS = ["Broadcast (behind golfer)", "Follow ball", "Side on", "Down the line, high", "Top down"]


class Replay3D(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = self.session = self.proj = None
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.clock = QtCore.QElapsedTimer()
        self.T = 0.0
        self.cam = None

    # Widgets
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.club = QtWidgets.QComboBox()
        self.club.addItems(list(ASSUMED["spin_rpm"]))
        self.level = QtWidgets.QComboBox()
        self.level.addItems(["pro", "intermediate", "beginner", "custom"])
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(0, 12000)
        self.spin.setSingleStep(100)
        self.spin.setSuffix(" rpm")
        self.tilt = QtWidgets.QDoubleSpinBox()
        self.tilt.setRange(-30, 30)
        self.tilt.setSingleStep(1)
        self.tilt.setSuffix("° axis")
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(SPEEDS)
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(CAMERAS)
        self.b_play = QtWidgets.QPushButton("▶ Play")
        self.b_play.setObjectName("primary")
        self.b_replay = QtWidgets.QPushButton("⟲ Replay")
        self.b_export = QtWidgets.QPushButton("Export GIF…")
        for lab, w in (
            ("Club", self.club),
            ("Player", self.level),
            ("Spin", self.spin),
            ("", self.tilt),
            ("Speed", self.speed),
            ("Camera", self.camera),
        ):
            if lab:
                row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.b_play)
        row.addWidget(self.b_replay)
        row.addWidget(self.b_export)
        lay.addLayout(row)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((112, 158, 205))  # sky
        lay.addWidget(self.view, 1)
        self.hud = QtWidgets.QLabel(self.view)
        self.hud.setStyleSheet(
            "background: rgba(10,14,20,190); color: white; padding: 8px 12px; border-radius: 6px;"
            "font-size: 10pt;"
        )
        self.hud.move(12, 12)
        self.legend = QtWidgets.QLabel(self.view)
        self.legend.setStyleSheet(
            "background: rgba(10,14,20,170); color: white; padding: 4px 10px; border-radius: 5px;"
            "font-size: 9pt;"
        )
        self.clock_lbl = QtWidgets.QLabel(self.view)
        self.clock_lbl.setStyleSheet(
            "background: rgba(10,14,20,170); color: #cfd6e0; padding: 3px 8px; border-radius: 5px;"
            "font-size: 9pt;"
        )

        self.club.currentIndexChanged.connect(self._profile_changed)
        self.level.currentIndexChanged.connect(self._profile_changed)
        self.spin.valueChanged.connect(self._spin_edited)
        self.tilt.valueChanged.connect(lambda *_: self._recompute())
        self.camera.currentIndexChanged.connect(lambda *_: self._place_camera(snap=True))
        self.b_play.clicked.connect(self.toggle)
        self.b_replay.clicked.connect(self.replay)
        self.b_export.clicked.connect(self.export_gif)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_overlays()

    def _place_overlays(self):
        self.hud.adjustSize()
        self.legend.adjustSize()
        self.clock_lbl.adjustSize()
        h = self.view.height()
        self.legend.move(12, h - self.legend.height() - 12)
        self.clock_lbl.move(
            self.view.width() - self.clock_lbl.width() - 12, h - self.clock_lbl.height() - 12
        )

    # Data
    def set_result(self, res, session):
        self.stop()
        self.res, self.session = res, session
        ok = res is not None and res.speed is not None and res.shifts
        golf = ok and res.ball == "golf"
        for w in (self.club, self.level):
            w.setEnabled(bool(golf))
        if not ok:
            self.view.clear()
            self.hud.setText("No trajectory for this take (rejected before the angle stage).")
            self._place_overlays()
            return
        self._set_spin_from_profile() if golf else self._set_spin(0)
        self._recompute()

    def _set_spin(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(int(v))
        self.spin.blockSignals(False)

    def _set_spin_from_profile(self):
        if self.level.currentText() == "custom":
            return
        self._set_spin(ASSUMED["spin_rpm"][self.club.currentText()][self.level.currentText()])

    def _profile_changed(self, *_):
        self._set_spin_from_profile()
        self._recompute()

    def _spin_edited(self, *_):
        self.level.blockSignals(True)
        self.level.setCurrentText("custom")
        self.level.blockSignals(False)
        self._recompute()

    def _recompute(self):
        result, session = self.res, self.session
        if result is None or result.speed is None:
            return
        x = result.shifts[result.slot_shift]["x"]
        # last tracked frame, s after launch
        tau_last = float(np.max(result._arrays["frames"]["t"]) - x[3])
        self.proj = project(
            x,
            result.ball,
            session.p0(result.ball),
            session.radar,
            tau_last,
            self.spin.value(),
            self.tilt.value(),
            session.tee_xy_m,
        )
        proj = self.proj
        self.t_all = np.r_[proj["t_meas"], proj["t_proj"]]
        self.p_all = np.r_[proj["meas"], proj["proj"]]
        self.n_meas = len(proj["t_meas"])
        self.tau_last = tau_last
        self._build_scene()
        self.T = 0.0
        self._draw(0.0)
        self._place_camera(snap=True)
        self._update_hud()

    # Scene
    def _label(self, text, xyz, style="color: white;"):
        """Text pinned to a world point, as a QLabel over the view so it shows in grabs."""
        lb = QtWidgets.QLabel(text, self.view)
        lb.setStyleSheet(
            "background: rgba(10,14,20,140); padding: 1px 5px; border-radius: 3px; font-size: 8pt; "
            + style
        )
        lb.adjustSize()
        lb.show()
        self.labels.append((lb, np.array(xyz, float)))
        return lb

    def _update_labels(self):
        vp = self.view.getViewport()
        M = self.view.projectionMatrix(vp, vp) * self.view.viewMatrix()
        dpr = self.view.devicePixelRatioF()
        w, h = self.view.width(), self.view.height()
        for lb, p in self.labels:
            q = M.map(QtGui.QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0))
            if q.w() <= 0 or not lb.text():
                lb.hide()
                continue
            nx, ny = q.x() / q.w(), q.y() / q.w()
            px, py = (nx + 1) / 2 * vp[2] / dpr, (1 - ny) / 2 * vp[3] / dpr
            if (
                not (-0.2 * w < px < 1.2 * w and -0.2 * h < py < 1.2 * h)
                or abs(nx) > 1.05
                or abs(ny) > 1.05
            ):
                lb.hide()
                continue
            lb.adjustSize()
            lb.move(int(px - lb.width() / 2), int(py - lb.height() / 2))
            lb.show()
        for o in (self.hud, self.legend, self.clock_lbl):
            o.raise_()

    def _build_scene(self):
        v, S, P = self.view, self.session, self.proj
        v.clear()
        for lb, _ in getattr(self, "labels", []):
            lb.deleteLater()
        self.labels = []
        length = max(80.0, P["carry_m"] * 1.15 + 20)
        width = max(60.0, abs(P["lateral_m"]) * 2 + 40)
        # ground: rough, fairway strip along the target line, grid every 10 m
        # rough, out to the horizon
        v.addItem(self._quad((-600, -800), (length + 600, 800), 0.0, (0.16, 0.36, 0.17, 1)))
        v.addItem(self._quad((-5, -14), (length, 14), 0.01, (0.22, 0.50, 0.22, 1)))
        grid = gl.GLGridItem(color=(255, 255, 255, 28))
        grid.setSize(length + 30, 2 * width)
        grid.setSpacing(10, 10)
        grid.translate((length - 30) / 2, 0, 0.02)
        v.addItem(grid)
        for yd in range(50, int(length / 0.9144) + 1, 50):
            xm = yd * 0.9144 + S.tee_xy_m[0]
            v.addItem(
                gl.GLLinePlotItem(
                    pos=np.array([[xm, -14, 0.03], [xm, 14, 0.03]]), color=(1, 1, 1, 0.35), width=1
                )
            )
            self._label("%d yd" % yd, (xm, 15.5, 0.05))
        # rig: radar, tee, net
        rz = S.radar_height_m
        box = gl.GLBoxItem(size=QtGui.QVector3D(0.12, 0.3, 0.12), color=(220, 220, 220, 255))
        box.translate(-0.06, -0.15, rz - 0.06)
        v.addItem(box)
        v.addItem(
            gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, rz - 0.06]]), color=(0.8, 0.8, 0.8, 1), width=2
            )
        )
        tee = np.array([[S.tee_xy_m[0], S.tee_xy_m[1], 0.03]])
        v.addItem(gl.GLScatterPlotItem(pos=tee, color=(1, 1, 1, 1), size=6))
        nx = S.net_x_m
        v.addItem(self._vquad(nx, (-2.0, 2.0), (0, 3.0), (1, 1, 1, 0.12)))
        self._label("net %.0f m" % nx, (nx, 0.0, 3.2))
        # tracer, ball, shadow, landing
        self.l_meas = gl.GLLinePlotItem(color=GREEN, width=5, antialias=True, mode="line_strip")
        self.l_proj = gl.GLLinePlotItem(color=RED, width=5, antialias=True, mode="line_strip")
        self.ball = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=11)
        self.shadow = gl.GLScatterPlotItem(color=(0, 0, 0, 0.45), size=9)
        self.land = gl.GLScatterPlotItem(color=(1, 0.85, 0.1, 1), size=14)
        self.land_lbl = self._label(
            "", tuple(P["landing"][:2]) + (3.0,), style="color: #ffd84a; font-weight: 600;"
        )
        for it in (self.l_meas, self.l_proj, self.shadow, self.ball, self.land):
            v.addItem(it)
        self.l_meas.setGLOptions("translucent")
        self.l_proj.setGLOptions("translucent")

    @staticmethod
    def _quad(p0, p1, z, color):
        (x0, y0), (x1, y1) = p0, p1
        verts = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
        m = gl.MeshData(vertexes=verts, faces=np.array([[0, 1, 2], [0, 2, 3]]))
        it = gl.GLMeshItem(meshdata=m, color=color, smooth=False, shader=None)
        it.setGLOptions("opaque" if color[3] >= 1 else "translucent")
        return it

    @staticmethod
    def _vquad(x, ys, zs, color):
        verts = np.array(
            [[x, ys[0], zs[0]], [x, ys[1], zs[0]], [x, ys[1], zs[1]], [x, ys[0], zs[1]]], float
        )
        m = gl.MeshData(vertexes=verts, faces=np.array([[0, 1, 2], [0, 2, 3]]))
        it = gl.GLMeshItem(meshdata=m, color=color, smooth=False)
        it.setGLOptions("translucent")
        return it

    # Animation
    def _ball_at(self, T):
        t, p = self.t_all, self.p_all
        if T >= t[-1]:
            return p[-1], len(t)
        i = int(np.searchsorted(t, T))
        if i == 0:
            return p[0], 1
        f = (T - t[i - 1]) / max(t[i] - t[i - 1], 1e-12)
        return p[i - 1] + f * (p[i] - p[i - 1]), i

    def _draw(self, T):
        ball, i = self._ball_at(T)
        n_meas = self.n_meas
        meas = (
            np.r_[self.p_all[: min(i, n_meas)], [ball]]
            if T <= self.tau_last
            else self.p_all[:n_meas]
        )
        self.l_meas.setData(pos=meas)
        if T > self.tau_last:
            self.l_proj.setData(
                pos=(
                    np.r_[self.p_all[n_meas:i], [ball]]
                    if i > n_meas
                    else np.array([self.p_all[n_meas - 1], ball])
                )
            )
        else:
            self.l_proj.setData(pos=np.zeros((0, 3)))
        self.ball.setData(pos=np.array([ball]))
        self.shadow.setData(pos=np.array([[ball[0], ball[1], 0.04]]))
        done = T >= self.t_all[-1]
        P = self.proj
        if done:
            self.land.setData(pos=np.array([[P["landing"][0], P["landing"][1], 0.06]]))
            self.land_lbl.setText("%.0f yd" % P["carry_yd"])
        else:
            self.land.setData(pos=np.zeros((0, 3)))
            self.land_lbl.setText("")
        phase = "measured" if T <= self.tau_last else "projected"
        self.clock_lbl.setText(
            "t = %.2f s · %s · %s"
            % (min(T, self.t_all[-1]), phase, self.speed.currentText().split(" ")[0])
        )
        self._place_overlays()
        return ball, done

    def _rate(self, T):
        choice = self.speed.currentIndex()
        if choice == 0:  # Auto: 0.1x over the measured part, ease to 0.6x
            if T <= self.tau_last:
                return 0.1
            return 0.1 + 0.5 * min(1.0, (T - self.tau_last) / 0.6)
        return [None, 0.1, 0.25, 0.5, 1.0][choice]

    def _tick(self):
        dt = self.clock.restart() / 1000.0
        self.T += min(dt, 0.05) * self._rate(self.T)
        ball, done = self._draw(self.T)
        self._place_camera(ball=ball)
        if self.T > self.t_all[-1] + 1.5 * self._rate(self.T):
            self.stop()

    def toggle(self):
        if self.proj is None:
            return
        if self.timer.isActive():
            self.stop()
        else:
            if self.T >= self.t_all[-1]:
                self.T = 0.0
            self.clock.start()
            self.timer.start()
            self.b_play.setText("❚❚ Pause")

    def stop(self):
        self.timer.stop()
        self.b_play.setText("▶ Play")

    def replay(self):
        if self.proj is None:
            return
        self.stop()
        self.T = 0.0
        self._draw(0.0)
        self._place_camera(snap=True)
        self.toggle()

    # Camera
    def _place_camera(self, ball=None, snap=False):
        if self.proj is None:
            return
        P, S = self.proj, self.session
        tee = np.array([S.tee_xy_m[0], S.tee_xy_m[1], 0.0])
        ball = np.asarray(ball if ball is not None else self._ball_at(self.T)[0])
        land = np.array(P["landing"])
        mid = 0.5 * (tee + land)
        heading = np.degrees(np.arctan2(land[1] - tee[1], land[0] - tee[0]))
        # each target is (look-at point, distance, elevation deg, azimuth deg)
        choice = self.camera.currentIndex()
        if choice == 0:  # broadcast: behind the golfer, pulling back as the ball goes out
            out = np.linalg.norm(ball - tee)
            target = (
                0.5 * (tee + ball) + np.array([0, 0, 0.3 * ball[2]]),
                6.0 + 1.05 * out,
                7.0 + 0.04 * out,
                180.0 + 0.3 * heading,
            )
        elif choice == 1:  # follow: behind and above the ball
            target = (ball, 9.0, 9.0, 180.0 + heading)
        elif choice == 2:  # side on, from the right of the target line
            target = (
                mid + np.array([0, 0, 0.5 * P["apex_m"]]),
                1.05 * P["carry_m"] + 20,
                4.0,
                -90.0,
            )
        elif choice == 3:  # down the line, high
            target = (mid, 0.85 * P["carry_m"] + 20, 22.0, 180.0 + heading)
        else:  # top down, downrange up the screen
            target = (mid, 1.9 * P["carry_m"] + 30, 89.0, 180.0)
        if snap or self.cam is None:
            self.cam = [np.array(target[0], float), *target[1:]]
        else:
            smoothing = 0.12
            self.cam[0] = self.cam[0] + smoothing * (np.asarray(target[0]) - self.cam[0])
            for k in (1, 2, 3):
                self.cam[k] = self.cam[k] + smoothing * (target[k] - self.cam[k])
        ctr, dist, el, az = self.cam
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(*map(float, ctr)),
            distance=float(dist),
            elevation=float(el),
            azimuth=float(az),
        )
        self._update_labels()

    def _update_hud(self):
        r, P = self.res, self.proj
        side = "L" if r.launch_az > 0 else "R"
        lateral = P["lateral_m"]
        warn = (
            ""
            if r.status == "ACCEPT"
            else "<tr><td colspan=3 style='color:#ff8a80'><b>REJECTED TAKE</b> — see banner</td></tr>"
        )
        self.hud.setText(
            "<table cellspacing=6>"
            + warn
            + "<tr><td><b>BALL SPEED</b></td><td>%.0f km/h</td><td style='color:#8fe39b'>measured</td></tr>"
            "<tr><td><b>LAUNCH</b></td><td>%.1f°%s</td><td style='color:#8fe39b'>measured</td></tr>"
            "<tr><td><b>DIRECTION</b></td><td>%.1f° %s</td><td style='color:#8fe39b'>measured</td></tr>"
            "<tr><td><b>SPIN</b></td><td>%d rpm</td><td style='color:#e8c547'>assumed</td></tr>"
            "<tr><td><b>CARRY</b></td><td>%.0f yd (%.0f m)</td><td style='color:#ff8a80'>projected</td></tr>"
            "<tr><td><b>APEX</b></td><td>%.1f m</td><td style='color:#ff8a80'>projected</td></tr>"
            "<tr><td><b>OFFLINE</b></td><td>%.1f m %s</td><td style='color:#ff8a80'>projected</td></tr>"
            "</table>"
            % (
                r.speed * 3.6,
                r.launch_el,
                "" if r.launch_el_measured else " (range only)",
                abs(r.launch_az),
                side,
                P["spin_rpm"],
                P["carry_yd"],
                P["carry_m"],
                P["apex_m"],
                abs(lateral),
                "L" if lateral > 0 else "R",
            )
        )
        self.legend.setText(
            "<span style='color:#40f259'>━━</span> measured&nbsp;&nbsp;"
            "<span style='color:#ff4033'>━━</span> projected"
        )
        self._place_overlays()

    # Export
    def export_gif(self, path=None, fps=25, width=960):
        """Render the replay at the current speed and camera to an animated GIF."""
        if self.proj is None:
            return None
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save replay", "replay.gif", "GIF (*.gif)"
            )
            if not path:
                return None
        from PIL import Image

        self.stop()
        frames, T, dt = [], 0.0, 1.0 / fps
        end = self.t_all[-1] + 1.0 * self._rate(self.t_all[-1])
        self.T = 0.0
        self._draw(0.0)
        self._place_camera(snap=True)
        app = QtWidgets.QApplication.instance()
        while T <= end:
            ball, _ = self._draw(T)
            self._place_camera(ball=ball)
            app.processEvents()
            img = self.view.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
            ptr = img.constBits()
            ptr.setsize(img.sizeInBytes())
            pixels = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[
                :, : img.width() * 3
            ]
            im = Image.fromarray(pixels.reshape(img.height(), img.width(), 3).copy())
            if im.width > width:
                im = im.resize((width, int(im.height * width / im.width)))
            frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
            T += dt * self._rate(T)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        self.T = T
        return path
