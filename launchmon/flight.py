"""Drag + gravity flight model and a small Levenberg-Marquardt solver, numpy only."""

import numpy as np

G = 9.81


class _Solution:
    pass


def least_squares(fun, x0, x_scale=None):
    """Levenberg-Marquardt with a forward-difference Jacobian. Returns .x, .fun and .jac."""
    x = np.array(x0, float)
    scale = np.array(x_scale if x_scale is not None else np.ones_like(x), float)
    damping = 1e-3
    residual = fun(x)
    cost = residual @ residual
    jac = None
    for _ in range(200):
        jac = np.empty((residual.size, x.size))
        for j in range(x.size):
            step = 1e-6 * scale[j]
            x_step = x.copy()
            x_step[j] += step
            jac[:, j] = (fun(x_step) - residual) / step
        normal = jac.T @ jac
        gradient = jac.T @ residual
        while True:
            dx = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), gradient)
            x_new = x + dx
            residual_new = fun(x_new)
            cost_new = residual_new @ residual_new
            if cost_new < cost:
                damping = max(damping / 3, 1e-9)
                break
            damping *= 5
            if damping > 1e8:
                break
        if damping > 1e8 or abs(cost - cost_new) < 1e-14 * max(cost, 1e-30):
            if cost_new < cost:
                x, residual, cost = x_new, residual_new, cost_new
            break
        x, residual, cost = x_new, residual_new, cost_new
    solution = _Solution()
    solution.x, solution.fun, solution.jac = x, residual, jac
    return solution


def _rk4(rhs, y0, t_eval, dt=2.5e-4):
    """Fixed-step fourth-order Runge-Kutta from t = 0, sampled at t_eval (any order)."""
    t_eval = np.asarray(t_eval, float)
    order = np.argsort(t_eval)
    y = np.array(y0, float)
    t = 0.0
    out = np.empty((len(y0), t_eval.size))
    for idx in order:
        t_target = t_eval[idx]
        while t < t_target - 1e-15:
            h = min(dt, t_target - t)
            k1 = rhs(t, y)
            k2 = rhs(t + h / 2, y + h / 2 * k1)
            k3 = rhs(t + h / 2, y + h / 2 * k2)
            k4 = rhs(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[:, idx] = y
    return out


def flight(v0, el, az, k, tt, p0):
    """World positions relative to the array centre at times tt after launch from p0.
    el and az in radians, az positive to the left."""
    v_launch = v0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def rhs(_, state):
        velocity = state[3:]
        return np.r_[velocity, -k * np.linalg.norm(velocity) * velocity + np.array([0, 0, -G])]

    tt = np.asarray(tt, float)
    return _rk4(rhs, np.r_[p0, v_launch], np.clip(tt, 0, None))[:3].T


def geo_fit(t, r, k, p0):
    """Range-only drag + gravity fit with azimuth fixed at 0: (v0, el, t_launch), sigma, rms."""

    def residuals(params):
        v0, el, t_launch = params
        return np.linalg.norm(flight(v0, el, 0.0, k, t - t_launch, p0), axis=1) - r

    v_guess = (r[-1] - r[0]) / (t[-1] - t[0]) / 0.8
    t_launch_guess = t[0] - (r[0] - np.linalg.norm(p0)) / (0.8 * v_guess)
    solution = least_squares(
        residuals, [v_guess, np.radians(15), t_launch_guess], x_scale=[10, 0.1, 0.01]
    )
    rms = np.sqrt(np.mean(solution.fun**2))
    cov = np.linalg.pinv(solution.jac.T @ solution.jac) * rms**2
    return solution.x, np.sqrt(np.diag(cov)), rms


def cubic_speed(t, r):
    """Speed from the t^2 coefficient of a cubic fit to r^2(t), the polynomial cross-check."""
    coef = np.polyfit(t - t[0], r**2, 3)
    return float(np.sqrt(max(coef[1], 0.0)))
