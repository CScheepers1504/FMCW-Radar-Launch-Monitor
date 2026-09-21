function out = fit_flight(tf, rMeas, azMeas, elMeas, opt)
% FIT_FLIGHT  Launch state by fitting the drag+gravity flight model to the measurements.
%
%  Output: speed, elev, azim, t0, elevWorld, azimWorld, sigma, sigmaScaled, residRMS_r,
%          condJ, nIter, converged, free, traj (3 x N predicted positions).


if nargin == 1 && ischar(tf) && strcmpi(tf, 'selftest')
    out = selftest(); return;
end

%% ---- options and validation -------------------------------------------
if nargin < 5, opt = struct(); end
opt = setdefault(opt, 'k',          []);
opt = setdefault(opt, 'ballType',   'golf');
opt = setdefault(opt, 'tiltDeg',    0);
opt = setdefault(opt, 'fixElevDeg', []);
opt = setdefault(opt, 'sigR',       0.01);
opt = setdefault(opt, 'sigAzDeg',   2.0);
opt = setdefault(opt, 'sigElDeg',   2.0);
opt = setdefault(opt, 'seed',       []);
opt = setdefault(opt, 'maxIter',    60);
opt = setdefault(opt, 'verbose',    true);
opt = setdefault(opt, 'nSub',       4);

assert(isfield(opt,'p0') && numel(opt.p0)==3, ...
    ['fit_flight: opt.p0 is REQUIRED -- the surveyed ball position at\n' ...
     'launch, [x y z] metres in the radar frame. Holding it fixed is what\n' ...
     'makes a 4-parameter fit possible over a 6 m arc. There is no sensible\n' ...
     'default; measure it once with a tape.']);

k = opt.k;
if isempty(k)
    % rho*Cd*A/2m, 1/m (Cd from Martin).
    switch lower(opt.ballType)
        case 'golf',     k = 4.84e-3;
        case 'cricket',  k = 6.23e-3;
        case 'football', k = 1.35e-2;
        otherwise, error('fit_flight: unknown ballType "%s"', opt.ballType);
    end
end

tf = tf(:);  rMeas = rMeas(:);
N  = numel(tf);
assert(numel(rMeas)==N, 'fit_flight: tf and rMeas length mismatch');
assert(N >= 8, 'fit_flight: only %d points, refusing to fit 4 parameters', N);

if nargin < 3 || isempty(azMeas), azMeas = nan(N,1); else, azMeas = azMeas(:); end
if nargin < 4 || isempty(elMeas), elMeas = nan(N,1); else, elMeas = elMeas(:); end

p0 = opt.p0(:);
tilt = opt.tiltDeg * pi/180;
g    = -9.81 * [sin(tilt); 0; cos(tilt)];

%% ---- seed --------------------------------------------------------------
% Seed: speed/direction from the data, t0 by extrapolating range back to |p0|.
% A short arc can have local minima, so the seed matters.
if isempty(opt.seed)
    pr    = polyfit(tf - tf(1), rMeas.^2, 2);
    sp0   = sqrt(max(pr(1), 1));              % t^2 coefficient
    az0   = median(azMeas(~isnan(azMeas)));  if isempty(az0)||isnan(az0), az0 = 0; end
    if ~isempty(opt.fixElevDeg)
        el0 = opt.fixElevDeg;
    else
        el0 = median(elMeas(~isnan(elMeas))); if isempty(el0)||isnan(el0), el0 = 10; end
    end
    t00 = tf(1) - (rMeas(1) - norm(p0)) / max(sp0, 1);
    seed = [sp0; el0; az0; t00];
else
    seed = opt.seed(:);
end

% Which parameters are free. Elevation drops out when it is pinned.
free = true(4,1);
if ~isempty(opt.fixElevDeg), seed(2) = opt.fixElevDeg; free(2) = false; end

sig = struct('r', opt.sigR, 'az', opt.sigAzDeg, 'el', opt.sigElDeg);
fun = @(th) residuals(th, tf, rMeas, azMeas, elMeas, p0, k, g, sig, opt.nSub);

%% ---- solve -------------------------------------------------------------
[theta, info] = lmsolve(fun, seed, free, opt.maxIter);

%% ---- report ------------------------------------------------------------
[res, pp] = fun(theta);
% Range residual in metres.
rRes   = rMeas - sqrt(sum(pp.^2, 1)).';
residR = sqrt(mean(rRes.^2));

% Covariance from column-scaled J (the raw columns span many orders of magnitude and
% J'J is numerically singular). sigmaScaled uses the achieved residual.
cnorm = sqrt(sum(info.J.^2, 1)).';
cnorm(cnorm < eps) = 1;
Jsc  = info.J ./ cnorm.';
Cf   = pinv(Jsc.' * Jsc) ./ (cnorm * cnorm.');
dof  = max(numel(res) - sum(free), 1);
Cs   = Cf * (sum(res.^2) / dof);
sfull = zeros(4,1); sscal = zeros(4,1);
sfull(free) = sqrt(max(diag(Cf), 0));
sscal(free) = sqrt(max(diag(Cs), 0));

% Launch angles in the world frame (elevation above the ground).
v_r = theta(1) * dirvec(theta(2), theta(3));
R   = [cos(tilt) 0 -sin(tilt); 0 1 0; sin(tilt) 0 cos(tilt)];
v_w = R * v_r;

out = struct();
out.speed       = theta(1);
out.elev        = theta(2);
out.azim        = theta(3);
out.t0          = theta(4);
out.elevWorld   = asin(max(min(v_w(3)/norm(v_w),1),-1)) * 180/pi;
out.azimWorld   = atan2(v_w(2), v_w(1)) * 180/pi;
out.sigma       = sfull;
out.sigmaScaled = sscal;
out.residRMS_r  = residR;
out.condJ       = cond(Jsc) * max(cnorm)/min(cnorm);
out.nIter       = info.nIter;
out.converged   = info.converged;
out.free        = free;
out.k           = k;
out.traj        = pp;
out.theta       = theta;

% Full covariance for propagate_flight: speed and elevation are strongly correlated.
% Pinned parameters get zero rows/columns.
Cfull = zeros(4); Cfull(free,free) = Cf;
Cscal = zeros(4); Cscal(free,free) = Cs;
out.cov         = Cfull;
out.covScaled   = Cscal;

% Carry the geometry forward so propagation uses the same tilt and p0.
out.p0          = p0;
out.tiltDeg     = opt.tiltDeg;

if opt.verbose
    fprintf('--- flight-model fit (drag k = %.3e 1/m, %s) ---\n', k, ...
            ternary(isempty(opt.k), opt.ballType, 'k supplied'));
    fprintf('  launch speed       : %7.3f m/s   +/- %.3f\n', out.speed, sscal(1));
    if free(2)
        fprintf('  elevation (world)  : %+7.2f deg   +/- %.2f\n', out.elevWorld, sscal(2));
    else
        fprintf('  elevation (world)  : %+7.2f deg   (HELD FIXED)\n', out.elevWorld);
    end
    fprintf('  azimuth   (world)  : %+7.2f deg   +/- %.2f\n', out.azimWorld, sscal(3));
    fprintf('  launch time        : %7.4f s     +/- %.4f  (%.1f ms before first frame)\n', ...
            out.t0, sscal(4), 1e3*(tf(1)-out.t0));
    fprintf('  range residual RMS : %7.1f cm\n', 100*residR);
    fprintf('  cond(J)            : %.2e\n', out.condJ);
    if ~info.converged
        fprintf('  ^^ DID NOT CONVERGE in %d iterations. Treat as unusable.\n', opt.nIter);
    end
    if out.condJ > 1e7
        fprintf(['  ^^ cond(J) > 1e7: at least one direction is not\n' ...
                 '     constrained by this data. Almost always elevation with\n' ...
                 '     no angle measurement -- set opt.fixElevDeg, or add the\n' ...
                 '     MIMO elevation channel. The number above is not a\n' ...
                 '     measurement.\n']);
    end
    if all(isnan(azMeas))
        fprintf('  note: no azimuth measurement supplied; azimuth rests on range alone.\n');
    end
end
end

%% ========================================================================
function [res, pp] = residuals(theta, tf, rMeas, azMeas, elMeas, p0, k, g, sig, nSub)
% Sigma-normalised residuals; NaN measurements contribute no rows.
pp = integrate_flight(theta, tf, p0, k, g, nSub);

rHat  = sqrt(sum(pp.^2, 1)).';
azHat = atan2(pp(2,:), pp(1,:)).' * 180/pi;
elHat = asin(max(min(pp(3,:).'./max(rHat,eps), 1), -1)) * 180/pi;

res = (rMeas - rHat) / sig.r;

m = ~isnan(azMeas);
if any(m), res = [res; wrap180(azMeas(m) - azHat(m)) / sig.az]; end
m = ~isnan(elMeas);
if any(m), res = [res; wrap180(elMeas(m) - elHat(m)) / sig.el]; end

% Diverged trial: large finite cost so LM rejects the step.
res(~isfinite(res)) = 1e6;
end

%% ------------------------------------------------------------------------
function pp = integrate_flight(theta, tf, p0, k, g, nSub)
% RK4 from t0, substepping between frame times.
v0 = theta(1) * dirvec(theta(2), theta(3));
t0 = theta(4);

y  = [p0; v0];
N  = numel(tf);
pp = zeros(3, N);
tc = t0;
for i = 1:N
    dt = tf(i) - tc;
    if dt < 0
        % Frame before launch: NaN, so the trial is rejected.
        pp(:,i) = NaN; tc = tf(i); continue;
    end
    h = dt / nSub;
    for s = 1:nSub
        y = rk4step(y, h, k, g);
    end
    pp(:,i) = y(1:3);
    tc = tf(i);
end
end

function y = rk4step(y, h, k, g)
k1 = deriv(y,          k, g);
k2 = deriv(y + h/2*k1, k, g);
k3 = deriv(y + h/2*k2, k, g);
k4 = deriv(y + h  *k3, k, g);
y  = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dy = deriv(y, k, g)
v  = y(4:6);
dy = [v; -k * norm(v) * v + g];
end

function u = dirvec(elDeg, azDeg)
el = elDeg*pi/180; az = azDeg*pi/180;
u  = [cos(el)*cos(az); cos(el)*sin(az); sin(el)];
end

function d = wrap180(d)
d = mod(d + 180, 360) - 180;
end

%% ------------------------------------------------------------------------
function [theta, info] = lmsolve(fun, theta, free, maxIter)
% Levenberg-Marquardt, forward-difference Jacobian 
% no Optimization Toolbox.
idx    = find(free);
n      = numel(idx);
lambda = 1e-3;
r      = fun(theta);
cost   = sum(r.^2);
converged = false;
it = 0;
J = zeros(numel(r), n);
for it = 1:maxIter
    % --- Jacobian ---
    for j = 1:n
        p  = idx(j);
        dp = 1e-6 * max(1, abs(theta(p)));
        th = theta; th(p) = th(p) + dp;
        J(:,j) = (fun(th) - r) / dp;
    end
    % Scale columns to unit norm: raw columns differ by orders of magnitude.
    cnorm = sqrt(sum(J.^2, 1)).';
    cnorm(cnorm < eps) = 1;
    Js = J ./ cnorm.';
    stepped = false;
    for tryLambda = 1:14
        % LM step via QR on [Js; sqrt(lambda)I], never forming J'J.
        dxs = [Js; sqrt(lambda)*eye(n)] \ [-r; zeros(n,1)];
        dx  = dxs ./ cnorm;
        th  = theta; th(idx) = th(idx) + dx;
        rn  = fun(th);  cn = sum(rn.^2);
        if isfinite(cn) && cn < cost
            if abs(cost - cn) < 1e-12 * max(cost,1), converged = true; end
            theta = th; r = rn; cost = cn;
            lambda = max(lambda/10, 1e-12);
            stepped = true;
            break;
        end
        lambda = lambda * 10;
    end
    if ~stepped || converged, converged = true; break; end
end
info = struct('J', J, 'nIter', it, 'converged', converged, 'cost', cost);
end

%% ------------------------------------------------------------------------
function s = setdefault(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end