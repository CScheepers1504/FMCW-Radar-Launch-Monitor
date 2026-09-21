function out = propagate_flight(fit, opt)
% PROPAGATE_FLIGHT  Forward-propagate a fitted launch state to landing, with an uncertainty band.
%
%   out = PROPAGATE_FLIGHT(fit, opt)
%   PROPAGATE_FLIGHT('selftest')     run against closed-form known truth
%
%  Modelled: drag a = -k|v|v, gravity, flat ground. Optional Magnus (off; needs spin and Cd).
%  Not modelled: wind (indoor net only), bounce/roll (landing = first contact).
%
%  Frame: world. Origin at the radar's ground projection, x downrange, y cross-range, z up.
%  The fit's radar-frame state is rotated in using the fit's own tilt.
%
%  Output
%    nominal  trajectory, landing point, carry, lateral, apex, descent angle, impact speed
%    band     landing samples, sigma and 95 % CI of carry and lateral, band along the flight
%    sens     per-source linearised contribution, m
%    net      crossing at the net plane (what the camera measures), with band
%    nFailed  draws that did not land within tMax


if nargin == 1 && ischar(fit) && strcmpi(fit, 'selftest')
    out = selftest(); return;
end

%% ---- options and validation -------------------------------------------
if nargin < 2, opt = struct(); end
opt = setdefault(opt, 'groundZ',  0);
opt = setdefault(opt, 'kFrac',    0.10);
opt = setdefault(opt, 'nDraws',   2000);
opt = setdefault(opt, 'seed',     12345);
opt = setdefault(opt, 'netRange', []);
opt = setdefault(opt, 'dt',       1e-3);
opt = setdefault(opt, 'tMax',     20);
opt = setdefault(opt, 'verbose',  true);
opt = setdefault(opt, 'magnus',   struct());
mg  = opt.magnus;
mg  = setdefault(mg, 'enable', false);
mg  = setdefault(mg, 'spinRpm', 0);
mg  = setdefault(mg, 'axis',   [0; 0; 1]);
mg  = setdefault(mg, 'Cd',     []);
mg  = setdefault(mg, 'radius', []);
mg  = setdefault(mg, 'clFun',  @(S) 1.5*S);

NL = char(10);

for f = {'speed','elev','azim','k','p0','tiltDeg'}
    assert(isfield(fit, f{1}), ...
        ['propagate_flight: fit.%s is missing. Pass the struct returned by' NL ...
         'fit_flight, which carries the geometry the propagation needs so' NL ...
         'that the tilt and p0 cannot disagree with the ones that were fitted.'], f{1});
end

% No default: launch height sets the carry directly.
assert(isfield(opt,'radarHeight') && ~isempty(opt.radarHeight) && ...
       ~isnan(opt.radarHeight), ...
    ['propagate_flight: opt.radarHeight is REQUIRED -- the radar phase' NL ...
     'centre above the ground, in metres. It sets the launch height and' NL ...
     'therefore the carry. Measure it on the rig; there is no safe default.']);

if isfield(fit, 'covScaled') && ~isempty(fit.covScaled)
    defCov = fit.covScaled;
elseif isfield(fit, 'sigmaScaled') && ~isempty(fit.sigmaScaled)
    % Old fit structs have marginals only; warn, since the correlations matter here.
    defCov = diag(fit.sigmaScaled(:).^2);
    if opt.verbose
        fprintf(['  NOTE: fit has no covariance, only marginal sigmas. The' NL ...
                 '  band below ignores speed/elevation correlation and is' NL ...
                 '  indicative only. Re-run fit_flight to get fit.covScaled.\n']);
    end
else
    defCov = zeros(4);
end
opt = setdefault(opt, 'cov', defCov);
C   = opt.cov;
assert(isequal(size(C), [4 4]), ...
    'propagate_flight: opt.cov must be 4x4 in [speed elev azim t0] order');

if mg.enable
    assert(~isempty(mg.Cd) && ~isempty(mg.radius), ...
        ['propagate_flight: Magnus needs opt.magnus.Cd and .radius.' NL ...
         'The lift scale is kL = CL*k/Cd and k = rho*Cd*A/2m cannot be' NL ...
         'split without Cd. Take Cd from Martin, the same source k came' NL ...
         'from -- do not reuse a number from another ball or another Re.']);
    assert(mg.spinRpm >= 0, 'propagate_flight: spinRpm must be non-negative');
end

k    = fit.k;
tilt = fit.tiltDeg * pi/180;
% Radar frame -> world, same rotation fit_flight uses for elevWorld.
R    = [cos(tilt) 0 -sin(tilt); 0 1 0; sin(tilt) 0 cos(tilt)];
hOff = [0; 0; opt.radarHeight];
gW   = [0; 0; -9.81];

p0w = R * fit.p0(:) + hOff;
assert(p0w(3) > opt.groundZ, ...
    ['propagate_flight: the surveyed launch point is at or below the' NL ...
     'landing plane (z = %.3f m, ground = %.3f m). Check radarHeight and' NL ...
     'the sign of tiltDeg -- a radar above the ball tilts DOWN, so its' NL ...
     'tiltDeg is negative.'], p0w(3), opt.groundZ);

%% ---- nominal trajectory ------------------------------------------------
theta0 = [fit.speed; fit.elev; fit.azim; 0];
nom    = run_one(theta0, k, p0w, R, gW, mg, opt, []);
% All draws measured against the nominal aim line.
aimRef = nom.aim;
assert(nom.landed, ...
    ['propagate_flight: the nominal shot did not land within tMax = %.1f s.' NL ...
     'Either the launch state is wrong or the ball is going nearly straight' NL ...
     'up. Check the fit before raising tMax.'], opt.tMax);

%% ---- Monte Carlo band --------------------------------------------------
% Sample launch state (correlated) and k (independent) together.
rngState = rng;                 % restore on exit -- do not clobber the caller
rng(opt.seed);
cleanupRng = onCleanup(@() rng(rngState));

L      = chol_psd(C);
draws  = theta0 + L * randn(4, opt.nDraws);
kDraw  = k * (1 + opt.kFrac * randn(1, opt.nDraws));
kDraw  = max(kDraw, 0);         % a negative drag would accelerate the ball

nD      = opt.nDraws;
landXY  = nan(2, nD);
carryD  = nan(1, nD);
latD    = nan(1, nD);
trajs   = cell(1, nD);
netHit  = nan(2, nD);
for i = 1:nD
    s = run_one(draws(:,i), kDraw(i), p0w, R, gW, mg, opt, aimRef);
    if ~s.landed, continue; end
    landXY(:,i) = s.pLand(1:2);
    carryD(i)   = s.carry;
    latD(i)     = s.lateral;
    trajs{i}    = s.p;
    if ~isempty(opt.netRange) && ~isnan(s.net(1))
        netHit(:,i) = s.net(1:2);
    end
end
ok      = ~isnan(carryD);
nFailed = nD - sum(ok);

band = struct();
band.pLand        = landXY(:, ok);
band.carry        = carryD(ok);
band.lateral      = latD(ok);
band.sigmaCarry   = std(band.carry);
band.sigmaLateral = std(band.lateral);
band.ci95Carry    = [pct(band.carry, 2.5),   pct(band.carry, 97.5)];
band.ci95Lateral  = [pct(band.lateral, 2.5), pct(band.lateral, 97.5)];

% Band along the flight on a common downrange grid (comparable across draws).
xEnd = pct(landXY(1, ok), 5);
xg   = linspace(p0w(1), xEnd, 200);
yAll = nan(sum(ok), numel(xg));
zAll = nan(sum(ok), numel(xg));
kept = find(ok);
for j = 1:numel(kept)
    P  = trajs{kept(j)};
    xv = P(1,:);
    % Skip draws with non-monotonic downrange rather than fail.
    good = [true, diff(xv) > 0];
    if nnz(good) < 4, continue; end
    yAll(j,:) = interp1(xv(good), P(2,good), xg, 'linear', NaN);
    zAll(j,:) = interp1(xv(good), P(3,good), xg, 'linear', NaN);
end
band.downrange  = xg;
band.lateralLo  = colpct(yAll, 2.5);
band.lateralMed = colpct(yAll, 50);
band.lateralHi  = colpct(yAll, 97.5);
band.heightLo   = colpct(zAll, 2.5);
band.heightHi   = colpct(zAll, 97.5);

%% ---- where the spread comes from ---------------------------------------
% One-at-a-time linearised contributions: which measurement to improve next.
% Does not sum to the Monte Carlo band (parameters are correlated).
names = {'speed','elevation','azimuth','t0','drag k'};
sig1  = [sqrt(max(diag(C),0)).', opt.kFrac * k];
sens  = struct('name', {names}, 'sigma', sig1, ...
               'dCarry', zeros(1,5), 'dLateral', zeros(1,5));
for j = 1:5
    if sig1(j) <= 0, continue; end
    th = theta0; kk = k;
    vals = [theta0(:); k];
    dp   = 1e-4 * max(1, abs(vals(j)));
    if j <= 4, th(j) = th(j) + dp; else, kk = kk + dp; end
    s = run_one(th, kk, p0w, R, gW, mg, opt, aimRef);
    if ~s.landed, continue; end
    sens.dCarry(j)   = abs(s.carry   - nom.carry)   / dp * sig1(j);
    sens.dLateral(j) = abs(s.lateral - nom.lateral) / dp * sig1(j);
end

%% ---- net crossing ------------------------------------------------------
net = struct();
if ~isempty(opt.netRange)
    net.nominal = nom.net;                      % [y z t] at the plane
    nOk         = ~isnan(netHit(1,:));
    net.y       = netHit(1, nOk);
    net.z       = netHit(2, nOk);
    net.sigmaY  = std(net.y);
    net.sigmaZ  = std(net.z);
    net.ci95Y   = [pct(net.y, 2.5), pct(net.y, 97.5)];
    net.ci95Z   = [pct(net.z, 2.5), pct(net.z, 97.5)];
end

out = struct('nominal', nom, 'band', band, 'sens', sens, 'net', net, ...
             'nFailed', nFailed, 'k', k, 'p0World', p0w, 'opt', opt);

%% ---- report ------------------------------------------------------------
if opt.verbose
    fprintf('--- forward propagation (drag k = %.3e 1/m, %d draws) ---\n', k, nD);
    fprintf('  launch (world)     : [%.2f %.2f %.2f] m, %.2f m/s at %+.2f deg\n', ...
            p0w, fit.speed, nom.elevWorld);
    fprintf('  flight time        : %7.3f s\n', nom.tLand);
    fprintf('  carry              : %7.3f m   +/- %.3f  (95%% %.2f to %.2f)\n', ...
            nom.carry, band.sigmaCarry, band.ci95Carry);
    fprintf('  lateral at landing : %+7.3f m   +/- %.3f  (95%% %+.2f to %+.2f)\n', ...
            nom.lateral, band.sigmaLateral, band.ci95Lateral);
    fprintf('  apex               : %7.3f m at %.2f m downrange\n', ...
            nom.apex, nom.apexRange);
    fprintf('  impact             : %7.2f m/s at %.1f deg descent\n', ...
            nom.impactSpeed, nom.descentDeg);
    if ~isempty(opt.netRange)
        fprintf('  at the net (%.2f m)  : y %+.3f +/- %.3f m, z %.3f +/- %.3f m\n', ...
                opt.netRange, net.nominal(1), net.sigmaY, net.nominal(2), net.sigmaZ);
    end
    fprintf('  where the spread comes from (1-sigma, linearised):\n');
    fprintf('      %-10s  %-10s  %8s  %8s\n', 'source', 'sigma', 'carry m', 'lat m');
    for j = 1:5
        fprintf('      %-10s  %-10.4g  %8.3f  %8.3f\n', ...
                names{j}, sig1(j), sens.dCarry(j), sens.dLateral(j));
    end
    fprintf(['      t0 must read 0.000 on both: shifting the launch instant\n' ...
             '      moves when the ball lands, not where.\n']);
    if nFailed > 0
        fprintf('  %d of %d draws did not land within %.1f s and were dropped.\n', ...
                nFailed, nD, opt.tMax);
    end
    if ~mg.enable
        fprintf(['  MAGNUS IS OFF. With any real sidespin the CENTRE of the\n' ...
                 '  lateral band above is wrong, not just its width. This is\n' ...
                 '  a stated limit, not a safe default.\n']);
    else
        fprintf(['  Magnus ON at %.0f rpm, CL from the DEFAULT nominal law.\n' ...
                 '  That law is provisional pending Bearman & Harvey -- do not\n' ...
                 '  quote a lateral number from it as measured.\n'], mg.spinRpm);
    end
    if isfield(fit,'condJ') && fit.condJ > 1e7
        fprintf(['  ^^ the fit reported cond(J) = %.1e. At least one launch\n' ...
                 '     direction is unconstrained, so this band is propagating\n' ...
                 '     an unmeasured parameter. Fix the fit first.\n'], fit.condJ);
    end
end
end

%% ========================================================================
function s = run_one(theta, k, p0w, R, gW, mg, opt, aimRef)
% One trajectory in the world frame; theta = radar-frame [speed elevDeg azimDeg t0].
% aimRef must stay the nominal aim for every draw, or azimuth uncertainty cancels and the
% lateral band reads zero. [] only for the nominal run.
if nargin < 8, aimRef = []; end
v0w = theta(1) * (R * dirvec(theta(2), theta(3)));

% Magnus scale kL = CL*k/Cd (shares rho*A/2m with k). CL depends on S = wR/|v|, so
% recompute each call.
if mg.enable && mg.spinRpm > 0
    om   = mg.spinRpm * 2*pi/60;
    axs  = mg.axis(:) / max(norm(mg.axis), eps);
    magP = struct('on', true, 'om', om, 'axs', axs, 'R', mg.radius, ...
                  'kOverCd', k / mg.Cd, 'clFun', mg.clFun);
else
    magP = struct('on', false, 'om', 0, 'axs', [0;0;1], 'R', 0, ...
                  'kOverCd', 0, 'clFun', @(S) 0);
end

h     = opt.dt;
nMax  = ceil(opt.tMax / h) + 2;
tv    = zeros(1, nMax);
pv    = zeros(3, nMax);
vv    = zeros(3, nMax);
y     = [p0w; v0w];
tv(1) = 0; pv(:,1) = y(1:3); vv(:,1) = y(4:6);
n     = 1;
landed = false;
for i = 2:nMax
    yPrev = y;
    y = rk4step(y, h, k, gW, magP);
    if ~all(isfinite(y)), break; end
    n = i; tv(n) = tv(n-1) + h; pv(:,n) = y(1:3); vv(:,n) = y(4:6);
    if y(3) <= opt.groundZ
        % Bisect the last step to land exactly on the plane (1 ms at 40 m/s = 4 cm).
        lo = 0; hi = h;
        for b = 1:60
            mid = 0.5*(lo + hi);
            ym  = rk4step(yPrev, mid, k, gW, magP);
            if ym(3) > opt.groundZ, lo = mid; else, hi = mid; end
        end
        y = rk4step(yPrev, hi, k, gW, magP);
        tv(n) = tv(n-1) + hi; pv(:,n) = y(1:3); vv(:,n) = y(4:6);
        landed = true;
        break;
    end
end

s = struct();
s.t = tv(1:n); s.p = pv(:,1:n); s.v = vv(:,1:n);
s.landed = landed;
s.elevWorld = asin(max(min(v0w(3)/max(norm(v0w),eps),1),-1)) * 180/pi;
if isempty(aimRef)
    ah    = atan2(v0w(2), v0w(1));
    s.aim = [cos(ah); sin(ah)];
else
    s.aim = aimRef(:);
end
if landed
    s.tLand  = tv(n) + theta(4);
    s.pLand  = pv(:,n);
    s.vLand  = vv(:,n);
    d        = s.pLand(1:2) - p0w(1:2);
    s.carry  = norm(d);
    % Cross-range is signed relative to the aim line, not boresight.
    s.lateral = -s.aim(2)*d(1) + s.aim(1)*d(2);
    [s.apex, ia] = max(pv(3,1:n));
    s.apexRange  = pv(1,ia) - p0w(1);
    s.impactSpeed = norm(s.vLand);
    s.descentDeg  = atan2(-s.vLand(3), norm(s.vLand(1:2))) * 180/pi;
else
    s.tLand = NaN; s.pLand = nan(3,1); s.vLand = nan(3,1);
    s.carry = NaN; s.lateral = NaN; s.apex = NaN; s.apexRange = NaN;
    s.impactSpeed = NaN; s.descentDeg = NaN;
end

% Crossing of the net plane x = netRange, interpolated on the bracketing step.
s.net = [NaN NaN NaN];
if ~isempty(opt.netRange) && n >= 2
    ix = find(pv(1,1:n-1) < opt.netRange & pv(1,2:n) >= opt.netRange, 1);
    if ~isempty(ix)
        f = (opt.netRange - pv(1,ix)) / (pv(1,ix+1) - pv(1,ix));
        s.net = [pv(2,ix) + f*(pv(2,ix+1) - pv(2,ix)), ...
                 pv(3,ix) + f*(pv(3,ix+1) - pv(3,ix)), ...
                 tv(ix)   + f*(tv(ix+1)   - tv(ix))];
    end
end
end

%% ------------------------------------------------------------------------
function y = rk4step(y, h, k, g, mag)
k1 = deriv(y,          k, g, mag);
k2 = deriv(y + h/2*k1, k, g, mag);
k3 = deriv(y + h/2*k2, k, g, mag);
k4 = deriv(y + h  *k3, k, g, mag);
y  = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dy = deriv(y, k, g, mag)
v  = y(4:6);
sp = norm(v);
a  = -k * sp * v + g;
if mag.on && sp > eps
    S  = mag.om * mag.R / sp;              % spin factor
    CL = mag.clFun(S);
    kL = CL * mag.kOverCd;                 % rho*CL*A/2m
    a  = a + kL * sp * cross(mag.axs, v);
end
dy = [v; a];
end

function u = dirvec(elDeg, azDeg)
el = elDeg*pi/180; az = azDeg*pi/180;
u  = [cos(el)*cos(az); cos(el)*sin(az); sin(el)];
end

%% ------------------------------------------------------------------------
function L = chol_psd(C)
% Covariance square root via eigendecomposition: chol fails on pinned (zero) rows.
C = (C + C.') / 2;
[V, D] = eig(C);
d = max(diag(D), 0);
L = V * diag(sqrt(d));
L = real(L);
end

function v = pct(x, p)
% Percentile without the Statistics Toolbox (linear interpolation).
x = sort(x(~isnan(x)));
n = numel(x);
if n == 0, v = NaN; return; end
if n == 1, v = x; return; end
pos = (p/100) * n - 0.5;
pos = min(max(pos, 0), n-1);
lo  = floor(pos); hi = min(lo+1, n-1);
v   = x(lo+1) + (pos - lo) * (x(hi+1) - x(lo+1));
end

function v = colpct(M, p)
v = nan(1, size(M,2));
for j = 1:size(M,2)
    v(j) = pct(M(:,j).', p);
end
end

function s = setdefault(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end
