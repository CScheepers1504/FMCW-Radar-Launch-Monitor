function ang = angle_history(ctx, opt)
% ANGLE_HISTORY  Per-frame azimuth and elevation from the MIMO virtual array.
%
%   ang = ANGLE_HISTORY(ctx, opt)
%

if nargin < 2, opt = struct(); end
opt = setdef(opt, 'snrMin',     15);
opt = setdef(opt, 'frames',     []);
opt = setdef(opt, 'unwrapElev', false);
% Chirp slot -> physical Tx. The estimator assumes slot 2 is the raised, central Tx;
% both cancellations above depend on it.
opt = setdef(opt, 'txOrder',    [1 2 3]);
opt = setdef(opt, 'verbose',    false);
if ~isfield(opt,'hyp') || isempty(opt.hyp)
    % Nominal AWR1843BOOST layout, lambda/2 units -- a hypothesis, tested in diag.
    opt.hyp = struct('rxX', [0 1 2 3], 'txX', [0 2 4], 'txY', [0 1 0]);
end

sel = opt.frames;
if isempty(sel)
    sel = find(ctx.valid & ctx.snrHat > opt.snrMin);
end
sel = sel(:);
assert(~isempty(sel), ...
    'angle_history: no frames above %.0f dB SNR -- lower opt.snrMin', opt.snrMin);

nSel     = numel(sel);
rxSlope  = nan(nSel, ctx.numTx);
txStep13 = nan(nSel, 1);
phiEl2   = nan(nSel, 1);        % the DOUBLED phase, kept for unwrapping
vrEst    = nan(nSel, 1);

% Radial velocity for the A2 Tx1->Tx3 step, from the (median-filtered) range history; Doppler folds.
rsm = movmedian(ctx.rHat, 9, 'omitnan');
for k = 1:nSel
    f = sel(k);
    if f > 1 && f < ctx.numFrames && ~isnan(rsm(f+1)) && ~isnan(rsm(f-1))
        vrEst(k) = (rsm(f+1) - rsm(f-1)) / (2*ctx.framePeriod);
    end
end
vrEst(isnan(vrEst)) = 0;

for k = 1:nSel
    f  = sel(k);
    rp = frame_rp_a(ctx, f);
    if strcmpi(ctx.clutterMode, 'complex'); rp = rp - ctx.clutterC; end

    [~, b] = min(abs(ctx.rangeAxis - ctx.rHat(f)));
    Z = reshape(rp(b, :, :), [ctx.numRx, ctx.numTx, ctx.numLoops]);
    Z = Z(:, opt.txOrder, :);

    dphiTx = 4*pi * vrEst(k) * ctx.chirpPeriod / ctx.wavelength;

    acc1 = zeros(1, ctx.numTx);  accStep = 0;  accEl = 0;
    for L = 1:ctx.numLoops
        z = Z(:, :, L);
        for tx = 1:ctx.numTx
            acc1(tx) = acc1(tx) + sum(z(2:end, tx) .* conj(z(1:end-1, tx)));
        end
        accStep = accStep + sum(z(:, 3) .* conj(z(:, 1))) * exp(-1j*2*dphiTx);
        accEl   = accEl   + sum(z(:, 2).^2 .* conj(z(:, 1)) .* conj(z(:, 3)));
    end
    rxSlope(k, :) = angle(acc1);
    txStep13(k)   = angle(accStep);
    phiEl2(k)     = angle(accEl);          % = 2*pi*v, wrapped to (-pi, pi]
end

%% ---- inversion ---------------------------------------------------------
sl = median(rxSlope, 2);                   % rad per Rx element = pi*u
u  = max(min(sl ./ pi, 1), -1);

if opt.unwrapElev
    % Unwrap the doubled phase, then halve.
    phiEl2u = unwrap(phiEl2);
else
    phiEl2u = phiEl2;
end
v = max(min(phiEl2u ./ (2*pi), 1), -1);

elDeg = asin(v) * 180/pi;
% Divide out cos(el): u is a direction cosine, not sin(az).
cosEl = sqrt(max(1 - v.^2, eps));
azDeg = asin(max(min(u ./ cosEl, 1), -1)) * 180/pi;

%% ---- diagnostics -------------------------------------------------------
dis = max(abs(rxSlope - sl), [], 2);

% Only use frames where D = txStep13/sl is well conditioned: |sl| away from zero (ratio is noise
% near boresight) and below pi/D (the step wraps beyond that, e.g. 14 deg of azimuth for D = 4).
wrapEdge = pi / max(abs(opt.hyp.txX(3) - opt.hyp.txX(1)), 1);
okD   = abs(sl) > 0.3 & abs(sl) < wrapEdge;
Dmeas = txStep13(okD) ./ sl(okD);
ccm   = corrcoef(u, phiEl2/2);

ang = struct();
ang.frames   = sel;
ang.t        = ctx.t_s(sel);
ang.azDeg    = azDeg;
ang.elDeg    = elDeg;
ang.u        = u;
ang.v        = v;
ang.rxSlope  = rxSlope;
ang.txStep13 = txStep13;
ang.phiEl    = phiEl2 / 2;
ang.vrEst    = vrEst;
ang.sdAz     = std(diff(azDeg), 'omitnan') / sqrt(2);
ang.sdEl     = std(diff(elDeg), 'omitnan') / sqrt(2);
ang.diag = struct( ...
    'txDisagree',   median(dis, 'omitnan'), ...
    'D',            median(Dmeas, 'omitnan'), ...
    'Diqr',         iqr(Dmeas(~isnan(Dmeas))), ...
    'Dn',           nnz(okD), ...
    'Dhyp',         opt.hyp.txX(3) - opt.hyp.txX(1), ...
    'corrAzEl',     ccm(1,2), ...
    'elAmbigDeg',   30, ...
    'elNearLimit',  abs(median(elDeg,'omitnan')) > 25, ...
    'unwrapped',    opt.unwrapElev, ...
    'txOrder',      opt.txOrder);

if opt.verbose, print_diag(ang, opt); end
end

%% ------------------------------------------------------------------------
function print_diag(ang, opt)
fprintf('\n================ MIMO ANGLE HISTORY ================\n');
fprintf('frames used: %d (SNR > %.0f dB)\n', numel(ang.frames), opt.snrMin);
fprintf('\nA1  Rx phase slope agreement between Tx:\n');
fprintf('     max disagreement %.3f rad (%.1f deg)\n', ...
        ang.diag.txDisagree, ang.diag.txDisagree*180/pi);
if ang.diag.txDisagree > 0.35
    fprintf(['     ^^ the three Tx disagree. Either the virtual-channel\n' ...
             '        mapping v = rx + nRx*(tx-1) is wrong, or the Tx are not\n' ...
             '        firing in the order chirpCfg claims. Nothing below is\n' ...
             '        meaningful until this is resolved.\n']);
end
fprintf('\nA2  Tx1->Tx3 offset in Rx spacings: D = %.2f, IQR %.2f  [hyp %.0f]\n', ...
        ang.diag.D, ang.diag.Diqr, ang.diag.Dhyp);
fprintf('     from %d of %d frames with |Rx slope| > 0.3 rad; modulo 2*pi, so a\n', ...
        ang.diag.Dn, numel(ang.frames));
fprintf('     check on the hypothesis, not a measurement.\n');
if ang.diag.Dn < 20
    fprintf(['     ^^ too few frames off boresight to say anything. The Tx\n' ...
             '        geometry is UNRESOLVED, not contradicted.\n']);
elseif abs(ang.diag.D - ang.diag.Dhyp) > 1
    fprintf(['     ^^ D disagrees with the hypothesis by more than one Rx\n' ...
             '        spacing. Either the layout is not the nominal one or the\n' ...
             '        step has wrapped. A3 is the check that actually gates the\n' ...
             '        elevation, but this should be explained before publishing\n' ...
             '        any angle result.\n']);
end
fprintf('\nA3  centred phase vs sin(azimuth): r = %+.3f\n', ang.diag.corrAzEl);
if abs(ang.diag.corrAzEl) > 0.5
    fprintf(['     ^^ elevation phase tracks azimuth, so Tx2 is NOT midway\n' ...
             '        between Tx1 and Tx3 in x. The cancellation the estimator\n' ...
             '        relies on does not hold. DO NOT feed this elevation to\n' ...
             '        fit_flight -- solve for x and y jointly first.\n']);
else
    fprintf('     weak -- consistent with Tx2 horizontally centred, as assumed.\n');
end
fprintf('\nB   angles over %d frames:\n', numel(ang.frames));
fprintf('     azimuth   %+6.2f deg median, IQR %.2f, frame-to-frame sd %.2f\n', ...
        median(ang.azDeg,'omitnan'), iqr(ang.azDeg), ang.sdAz);
fprintf('     elevation %+6.2f deg median, IQR %.2f, frame-to-frame sd %.2f\n', ...
        median(ang.elDeg,'omitnan'), iqr(ang.elDeg), ang.sdEl);
fprintf(['     The frame-to-frame sd is the number that matters: about\n' ...
         '     2 deg/frame on elevation gives sigma_elev 0.38 deg after the fit.\n']);
if ang.sdEl > 4
    fprintf('     ^^ %.1f deg/frame is well outside that. Expect the fit to be loose.\n', ang.sdEl);
end
if ang.diag.elNearLimit && ~ang.diag.unwrapped
    fprintf(['     ^^ elevation is within 5 deg of the +/- 30 deg wrap. It may\n' ...
             '        already have folded. Try opt.unwrapElev, and check the\n' ...
             '        result against the radar tilt.\n']);
end
fprintf('====================================================\n');
end

%% ------------------------------------------------------------------------
function rp = frame_rp_a(ctx, f)
i0    = (f-1)*ctx.bytesPerFrame/2;
chunk = double(ctx.raw(i0 + (1:ctx.bytesPerFrame/2)));
switch lower(ctx.deinterleave)
    case 'lvds4'
        gq = reshape(chunk, 4, []);
        cplx = reshape([gq(1,:) + 1j*gq(3,:); gq(2,:) + 1j*gq(4,:)], [], 1);
    case 'iq2'
        cplx = chunk(1:2:end) + 1j*chunk(2:2:end);
    otherwise
        error('angle_history: unknown de-interleave mode "%s"', ctx.deinterleave);
end
if ctx.doConj; cplx = conj(cplx); end
assert(numel(cplx) == ctx.samplesPerFrame, ...
    'angle_history: frame %d unpacked to %d samples, expected %d', ...
    f, numel(cplx), ctx.samplesPerFrame);
cube = reshape(reshape(cplx, [ctx.numAdcSamples, ctx.numRx, ctx.numTx, ctx.numLoops]), ...
               [ctx.numAdcSamples, ctx.numVirtual, ctx.numLoops]);
rp   = fft(cube .* ctx.wRange, ctx.numAdcSamples, 1);
end

function s = setdef(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end
