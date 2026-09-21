% =========================================================================
%  FMCW PIPELINE:  raw ADC cube  ->  range history r(t)  ->  r^2(t) fit
%  =========================================================================
clear; clc; close all;

% Newline for multi-line assert and error messages
NL = char(10);  %#ok<CHARTEN>

%% 1. INPUT AND TUNABLE PARAMETERS
% -------------------------------------------------------------------------
sessionDir = 'C:\Users\Chris Scheepers\Desktop\results\2026-09-09_14-30-49_rugby-150m-100m-static-20-left';

% -------------------------------------------------------------
% Range gate -> Needs to change when dynamic detection is implemented. For now, it is hardcoded to the range of the target.
params.rangeGate = [0.2 2.0];   % [min max] metres for r^2 fit

params.fitWindow       = [2.0 2.8];          % [t0 t1] seconds, 
params.fitDeg          = 2;                % polynomial order for r^2(t) fit

% --- flight-model launch-state fit  --------------------------

% so leave this off: it is for real ball captures with a measured tee.
params.flight.enable     = false;
params.flight.p0         = [];      % [x y z] m, ball at launch, RADAR frame
params.flight.ballType   = 'golf';  % 'golf' | 'cricket' | 'football'
params.flight.k          = [];      % drag 1/m; [] = from ballType
% Radar tilt, degrees, UP-POSITIVE
params.flight.tiltDeg    = NaN;     
params.flight.fixElevDeg = [];      % deg, or [] to solve for it
params.flight.sigR       = 0.01;    % m,   measured 6.68 mm overall in sim
params.flight.sigAzDeg   = 2.0;     % deg
params.flight.sigElDeg   = 2.0;     % deg
% SNR floor for the angle extraction. Frames below this contribute range only.
params.flight.angSnrMin  = 15;      % dB
% leave this off unless shot actually folds.
params.flight.unwrapElev = false;

% --- forward propagation ------------------------------------
% Runs only after a successful flight-model fit
params.propagate.enable = false;
% Radar phase centre above ground, metres. Sets the launch height and
% initial conditions for the propagation.
params.propagate.radarHeight = NaN;  % Downrange distance to the net, metres.

% landing point. [] to skip.
params.propagate.netRange    = [];

% 1-sigma fractional ignorance in the drag parameter -> bounded by how well Cd is
% known, 10 % is an assumption.
params.propagate.kFrac       = 0.10;
params.propagate.nDraws      = 2000;

% --- spin ---------------------------------------------------
% Micro-Doppler spread on the ball's range bin -> current setup does not measure it well.
params.spin.enable = false;

% Where the return comes from, which sets the RMS-to-omega*R conversion.
% 'ring' or 'sphere'
params.spin.shape  = 'sphere';

% sin of the angle between the spin axis and the line of sight. 
% 1 assumes pure backspin with the radar in the flight plane
params.spin.sinAlpha = 1;
params.spin.snrMin   = 15;      % dB, higher than the detection floor

% --- clutter -------------------------------------------------------------
%   'complex'    : two-pass. Mean COMPLEX range profile per (bin, antenna,
%                  chirp) subtracted before detection. Cancels statics
%                  properly because it removes amplitude AND phase. Use this
%                  for a MOVING target.
%   'background' : subtract the mean complex profile of a separate
%                  capture recorded with the target removed. Use this for a static
%                  target 
%   'median'     : per-bin median of POWER across frames. 
%
%   'none'       : diagnostic only.
%
params.clutter         = 'none';

% Path to the target-free/ background capture. Required when clutter = 'background'.
params.bgSession       = '';

% Range bins skipped at the near edge when fitting alpha. Tx->Rx leakage
% dominates the first few bins 
params.bgLeakBins      = 3;

% --- signal processing ---------------------------------------------------
params.deinterleave    = 'lvds4';  % AWR1843 2-lane packing: I0 I1 Q0 Q1
params.conjugate       = [];       % [] = auto from adcbufCfg.SampleSwap
params.windowType      = 'hann';
% --- how the 12 virtual channels and the loops are combined ---------------
%   'noncoherent' sum of |x|^2 over everything. 
%   'coherent'    angle FFT across Rx, Doppler FFT across loops, take the
%                 strongest cell. 
params.integrate       = 'coherent';
params.angNFFT         = 64;       % zero-pad: avoids straddling loss only
params.dopNFFT         = 16;       % zero-pad: interpolates, adds no resolution
params.pfa             = 1e-4;
params.cfar.guard      = 2;
params.cfar.train      = 8;

%   'ca' cell-averaging, 'os' order-statistic.
params.cfar.type       = 'os';
params.cfar.osK        = [];       % [] = round(0.75 * 2*train)

% --- rejecting things that are not a ball --------------------------------
% Any bin that detects in more than this fraction of frames is assumed to be a
% fixture, and is masked out of the range history before the r^2 fit.
params.persistFrac     = [];

% A fixture is not one bin wide -> widens the mask by this many bins on each side.
params.persistDilate   = 2;

% Pick the range history by continuity instead of brightest in the frame
% OFF by default
params.track.enable    = false;
params.track.vmax      = 80;       % m/s, the fastest step the path may take

% --- diagnostic frames only, no detection path ----------------------

params.diagFrame       = [];       % [] = best-SNR detected frame in the fit window
params.diagQuietFrame  = [];       % [] = representative target-free frame
params.diagSkipFrames  = 5;        % never diagnose on the first N frames
params.dopplerNFFT     = 64;       % zero-pad: interpolates, adds no resolution
% Within-frame MTI (subtracting the slow-time mean) nulls zero Doppler
% The coherent across-frame cancellation already removes the
% statics, and does it properly. Leave this false.
params.rdRemoveStatic  = false;

%% 2. CONFIGURATION FROM THE CAPTURE ITSELF
% -------------------------------------------------------------------------
metaFile = fullfile(sessionDir, 'meta.json');
binFile  = fullfile(sessionDir, 'frames.bin');
assert(isfile(metaFile), ['meta.json not found in %s\n' ...
    'It carries the profile actually used. Pull it before analysing.'], sessionDir);
assert(isfile(binFile), 'frames.bin not found in %s', sessionDir);

meta = jsondecode(fileread(metaFile));
cfg  = meta.config;
dims = meta.dims;

c = 299792458;

numAdcSamples  = double(dims.n_samples);
numRx          = double(dims.n_rx);
numTx          = double(dims.n_tx);
numLoops       = double(dims.n_loops);
numVirtual     = numRx * numTx;
chirpsPerFrame = numTx * numLoops;

sampleRate = cfg.profileCfg.digOutSampleRate * 1e3;
freqSlope  = cfg.profileCfg.freqSlopeConst  * 1e12;
startFreq  = cfg.profileCfg.startFreq * 1e9;
chirpPeriod = (cfg.profileCfg.idleTime + cfg.profileCfg.rampEndTime) * 1e-6;
framePeriod = cfg.frameCfg.framePeriod * 1e-3;
pri         = numTx * chirpPeriod;

bandwidth   = freqSlope * (numAdcSamples / sampleRate);   % SAMPLED bandwidth
wavelength  = c / (startFreq + bandwidth/2);              % centre, not start
rangeRes    = c / (2 * bandwidth);
maxRange    = sampleRate * c / (2 * freqSlope);
maxVelocity = wavelength / (4 * pri);
velocityRes = wavelength / (2 * numLoops * pri);

rangeAxis   = (0:numAdcSamples-1).' * (maxRange / numAdcSamples);
dopplerAxis = (-numLoops/2 : numLoops/2-1).' * (2*maxVelocity / numLoops);

fprintf('================ CAPTURE =================\n');
fprintf('session      : %s\n', meta.session);
fprintf('profile      : %s\n', meta.config_path);
fprintf('channels     : %d Tx x %d Rx -> %d virtual\n', numTx, numRx, numVirtual);
fprintf('sampled BW   : %.0f MHz -> range res %.2f cm, max range %.2f m\n', ...
        bandwidth/1e6, rangeRes*100, maxRange);
fprintf('doppler      : +/-%.2f m/s in %d bins of %.2f m/s\n', ...
        maxVelocity, numLoops, velocityRes);
fprintf('frame rate   : %.1f Hz\n', 1/framePeriod);

%% 3. READ THE RAW ADC STREAM
% -------------------------------------------------------------------------
info = dir(binFile);
bytesPerFrame = double(dims.frame_bytes);
% --- integrity check -----------------------------
% modulo check to see if actual frame size is consistent with expected size
if isfield(meta, 'capture')
    cap = meta.capture;
    if isfield(cap, 'size_mismatches') && cap.size_mismatches > 0
        error(['recorder reported size_mismatches = %d of %d frames.\n' ...
               'The radar was running a DIFFERENT profile from the one\n' ...
               'meta.json describes. This capture is not usable -- re-record\n' ...
               'with session.sh --restart.'], cap.size_mismatches, cap.frames);
    end
    if isfield(cap, 'frames')
        expBytes = double(cap.frames) * bytesPerFrame;
        assert(expBytes == info.bytes, ...
           ['frames.bin is %d B, but meta.json says %d frames x %d B/frame\n' ...
            '= %d B. The recorded frame size does not match the profile.'], ...
            info.bytes, cap.frames, bytesPerFrame, expBytes);
    end
end
assert(mod(info.bytes, bytesPerFrame) == 0, ...
   ['frames.bin is %d B, not a whole multiple of the %d B/frame in ' ...
    'meta.json. Config and data disagree -- stop.'], info.bytes, bytesPerFrame);
numFrames = info.bytes / bytesPerFrame;
fprintf('frames.bin   : %d frames, %.2f s\n', numFrames, numFrames*framePeriod);

fid = fopen(binFile, 'r');
raw = fread(fid, Inf, 'int16=>int16');
fclose(fid);
samplesPerFrame = numAdcSamples * numRx * chirpsPerFrame;

% ADC utilisation and clipping check
adcPk   = double(max(abs(raw)));
adcClip = sum(abs(raw) >= 32767);
fprintf('ADC          : peak %d/32767 (%.1f%% FS), %.1f dB headroom, %d clipped\n', ...
        adcPk, 100*adcPk/32767, 20*log10(32767/max(adcPk,1)), adcClip);
if adcClip > 0
    fprintf(['  ^^ CLIPPING. Phase is destroyed where it clips -- angle and\n' ...
             '     coherent integration are both invalid. Lower rxGain.\n']);
elseif adcPk < 0.25*32767
    fprintf(['  ^^ under 25%% of full scale. rxGain (currently %d dB, max 52)\n' ...
             '     has room; raise it and re-record to see if SNR follows.\n'], ...
            cfg.profileCfg.rxGain);
end

if isempty(params.conjugate)
    doConj = (cfg.adcbufCfg.SampleSwap == 1);
    fprintf('I/Q order    : SampleSwap=%d -> conjugate=%d (auto)\n', ...
            cfg.adcbufCfg.SampleSwap, doConj);
else
    doConj = params.conjugate;
    fprintf('I/Q order    : conjugate=%d (forced)\n', doConj);
end

wRange = get_window(params.windowType, numAdcSamples);
wRange = wRange / sqrt(mean(wRange.^2));

%% 4. PASS 1 -- COHERENT CLUTTER ESTIMATE
% -------------------------------------------------------------------------
% The mean complex range profile per bin, antenna and chirp. 
clutterC   = complex(zeros(numAdcSamples, numVirtual, numLoops));
useClutter = any(strcmpi(params.clutter, {'complex', 'background'}));

if useClutter
    % The data's own mean complex range profile. 
    % 'complex' subtracts it directly 
    % 'background' needs it as the thing the its fitted to match

    selfC = complex(zeros(numAdcSamples, numVirtual, numLoops));
    for f = 1:numFrames
        selfC = selfC + frame_rp(raw, f, bytesPerFrame, params.deinterleave, ...
            doConj, numAdcSamples, numRx, numTx, numLoops, numVirtual, ...
            samplesPerFrame, wRange);
    end
    selfC = selfC / numFrames;
end

if strcmpi(params.clutter, 'complex')
    clutterC = selfC;

elseif strcmpi(params.clutter, 'background')
    assert(~isempty(params.bgSession), ['params.clutter = ''background'' needs '  ...
        'params.bgSession set to a capture' NL 'recorded with the target removed ' ...
        'and the profile unchanged.']);

    ref = struct('n_samples', numAdcSamples, 'n_rx', numRx, 'n_tx', numTx, ...
                 'n_loops',   numLoops,      'frame_bytes', bytesPerFrame);
    bgC = bg_mean_profile(params.bgSession, ref, params, doConj, wRange, ...
                          numVirtual, samplesPerFrame, meta);

    % --- fit one complex scalar per (virtual antenna, chirp) --------------
    % Fitted outside the range gate, so a static target inside the gate
    % cannot pull the calibration onto itself.
    bgFit = true(numAdcSamples, 1);
    bgFit(1:min(params.bgLeakBins, numAdcSamples)) = false;
    bgFit(rangeAxis >= params.rangeGate(1) & ...
          rangeAxis <= params.rangeGate(2)) = false;
    nBgFit = nnz(bgFit);
    assert(nBgFit >= 8, ['background: only %d range bins left to fit the phase ' ...
        'offset on.' NL 'The gate [%.2f %.2f] m plus %d leakage bins covers ' ...
        'almost the whole profile.' NL 'Narrow params.rangeGate.'], nBgFit, ...
        params.rangeGate(1), params.rangeGate(2), params.bgLeakBins);

    bFit    = bgC(bgFit, :, :);
    tFit    = selfC(bgFit, :, :);
    bgAlpha = sum(conj(bFit) .* tFit, 1) ./ max(sum(abs(bFit).^2, 1), realmin);

    clutterC = bgAlpha .* bgC;

    % --- report what the subtraction actually achieved --------------------
    pBefore  = sum(abs(tFit).^2, 'all');
    pAfter   = sum(abs(tFit - bgAlpha .* bFit).^2, 'all');
    pNaive   = sum(abs(tFit - bFit).^2, 'all');
    cancelDB = 10*log10(pBefore / max(pAfter, realmin));
    naiveDB  = 10*log10(pBefore / max(pNaive, realmin));
    magAlpha = abs(bgAlpha(:));
    phAlpha  = rad2deg(angle(bgAlpha(:)));

    fprintf('  alpha      : |a| %.3f to %.3f, phase %+.1f to %+.1f deg, %d channels\n', ...
            min(magAlpha), max(magAlpha), min(phAlpha), max(phAlpha), numel(magAlpha));
    fprintf('  cancels    : %.1f dB fitted, %.1f dB naive, over %d fit bins\n', ...
            cancelDB, naiveDB, nBgFit);
    if cancelDB < 3
        fprintf(['  ^^ under 3 dB. The two captures are NOT mutually coherent:\n' ...
                 '     the synthesiser did not come back with the same phase\n' ...
                 '     reference. Do not use this pair. Re-record back to back\n' ...
                 '     without restarting the driver, or use clutter = ''none''.\n']);
    elseif cancelDB < naiveDB + 1
        fprintf(['  ^^ the phase fit bought under 1 dB, so the two captures were\n' ...
                 '     already aligned. Worth recording: it means restart phase\n' ...
                 '     is reproducible on this radar.\n']);
    end

    % --- per-bin cancellation inside the gate ----------------------------
  
    gateMask  = rangeAxis >= params.rangeGate(1) & rangeAxis <= params.rangeGate(2);
    pSelfBin  = sum(abs(selfC).^2, [2 3]);
    pResBin   = sum(abs(selfC - clutterC).^2, [2 3]);
    cancelBin = 10*log10(pSelfBin ./ max(pResBin, realmin));

    gIdx = find(gateMask);
    gCan = cancelBin(gIdx);
    poor = gCan < 3;
    fprintf('  in gate    : median %.1f dB, worst %.1f dB, %d of %d bins under 3 dB\n', ...
            median(gCan), min(gCan), nnz(poor), numel(gCan));

    % Peak-to-median contrast
    [pkBefore, iB] = max(pSelfBin(gIdx));
    [pkAfter,  iA] = max(pResBin(gIdx));
    conBefore = 10*log10(pkBefore / median(pSelfBin(gIdx)));
    conAfter  = 10*log10(pkAfter  / median(pResBin(gIdx)));
    fprintf('  contrast   : %.1f dB at %.2f m before, %.1f dB at %.2f m after\n', ...
            conBefore, rangeAxis(gIdx(iB)), conAfter, rangeAxis(gIdx(iA)));

    % Longest contiguous run of poorly-cancelling bins, in bins and metres.
    runLen = 0; bestRun = 0; bestEnd = 0;
    for i = 1:numel(poor)
        if poor(i)
            runLen = runLen + 1;
            if runLen > bestRun; bestRun = runLen; bestEnd = i; end
        else
            runLen = 0;
        end
    end
    cfarWin = 2*(params.cfar.guard + params.cfar.train) + 1;
    if bestRun > 0
        b1 = gIdx(bestEnd - bestRun + 1); b2 = gIdx(bestEnd);
        fprintf('  worst run  : %d bins, %.2f-%.2f m (CFAR window is %d bins)\n', ...
                bestRun, rangeAxis(b1), rangeAxis(b2), cfarWin);
        if bestRun >= cfarWin
            fprintf(['  ^^ that run is WIDER than the CFAR training window, so the\n' ...
                     '     threshold in those bins is set by the uncancelled residual\n' ...
                     '     itself. Expect no detections there whatever the target SNR.\n' ...
                     '     The two captures differ by more than the target: something\n' ...
                     '     else in the scene moved. Re-record the background without\n' ...
                     '     disturbing anything but the ball.\n']);
        end
    end
end

%% 5. PASS 2 -- RANGE PROCESSING AND CLUTTER CANCELLATION
% -------------------------------------------------------------------------
rangeTime = zeros(numAdcSamples, numFrames);
rawTime   = zeros(numAdcSamples, numFrames);   % before cancellation, for the plot
angIdx    = ones(numAdcSamples, numFrames, 'uint16');

nAng = max(params.angNFFT, numRx);
nDop = max(params.dopNFFT, numLoops);
if strcmpi(params.integrate, 'noncoherent'); nAng = 1; nDop = 1; end

for f = 1:numFrames
    rp = frame_rp(raw, f, bytesPerFrame, params.deinterleave, doConj, ...
        numAdcSamples, numRx, numTx, numLoops, numVirtual, samplesPerFrame, wRange);
    rawTime(:, f) = integrate_cube(rp, params.integrate, nAng, nDop, ...
                                   numRx, numTx, numLoops);

    if useClutter; rp = rp - clutterC; end
    [rangeTime(:, f), angIdx(:, f)] = integrate_cube(rp, params.integrate, ...
                                   nAng, nDop, numRx, numTx, numLoops);
end
fprintf('integration  : %s', lower(params.integrate));
if ~strcmpi(params.integrate, 'noncoherent')
    fprintf(' (%d angle x %d doppler cells, %d Tx summed in power)', ...
            nAng, nDop, numTx);
end
fprintf('\n');

switch lower(params.clutter)
    case 'complex';    residual = rangeTime;                    % already cancelled
    case 'background'; residual = rangeTime;                    % already cancelled
    case 'median';     residual = max(rangeTime - median(rangeTime,2), 0);
    case 'none';       residual = rangeTime;
    otherwise;         error('unknown clutter mode "%s"', params.clutter);
end

%% 6. 1D CFAR ACROSS RANGE, IN LINEAR POWER
% -------------------------------------------------------------------------
% Two noise estimators, both solved at the SAME design P_fa so that swapping
% between them is not a threshold change in disguise.
%
%   'ca'  cell averaging.  alpha = N*(P_fa^(-1/N) - 1).  The estimator Cross
%         (2025) used to land on alpha = 10, and the right default in clean
%         noise.
%   'os'  order statistic: sort the training cells, take the k-th (k = 3N/4).
%         Immune to a strong interferer sitting INSIDE the training window,
%         which is exactly what a stationary person in the range gate does to
%         every bin within +/-(guard+train) of them.
%         P_fa = prod_{i=0}^{k-1} (N-i)/(N-i+alpha), solved by bisection.
%

g = params.cfar.guard;  t = params.cfar.train;
nTrain = 2*t;
kOS = params.cfar.osK;
if isempty(kOS); kOS = round(0.75 * nTrain); end

switch lower(params.cfar.type)
    case 'ca'
        alpha = nTrain * (params.pfa^(-1/nTrain) - 1);
    case 'os'
        pfaOS = @(a) prod((nTrain-(0:kOS-1)) ./ (nTrain-(0:kOS-1) + a));
        lo = 1e-6; hi = 1e6;
        for it = 1:200                      % bisect in log space
            mid = sqrt(lo*hi);
            if pfaOS(mid) > params.pfa; lo = mid; else; hi = mid; end
        end
        alpha = sqrt(lo*hi);
    otherwise
        error('unknown params.cfar.type "%s" (expected ''ca'' or ''os'')', ...
              params.cfar.type);
end

gateLo = max(find(rangeAxis >= params.rangeGate(1), 1), g + t + 1);
gateHi = min(find(rangeAxis <= params.rangeGate(2), 1, 'last'), numAdcSamples - g - t);
assert(~isempty(gateLo) && ~isempty(gateHi) && gateHi > gateLo, ...
   ['range gate [%.2f %.2f] m leaves no usable bins (res %.3f m, max %.2f m, ' ...
    'CFAR needs %d bins of margin each side).'], params.rangeGate(1), ...
    params.rangeGate(2), rangeRes, maxRange, g+t);
fprintf('CFAR         : %s, alpha %.2f (%.1f dB), gate %.2f-%.2f m (bins %d-%d)\n', ...
        upper(params.cfar.type), alpha, 10*log10(alpha), ...
        rangeAxis(gateLo), rangeAxis(gateHi), gateLo, gateHi);

% Keep the noise estimate for the SNR calculation, and the detection mask for the range history.
detections = false(numAdcSamples, numFrames);
noiseEst   = nan(numAdcSamples, numFrames);
useOS      = strcmpi(params.cfar.type, 'os');
for f = 1:numFrames
    col = residual(:, f);
    for r = gateLo:gateHi
        win = col(r-g-t : r+g+t);
        tr  = [win(1:t); win(end-t+1:end)];
        if useOS
            srt = sort(tr);  nz = srt(kOS);
        else
            nz = mean(tr);
        end
        noiseEst(r, f)   = nz;
        detections(r, f) = col(r) > alpha * nz;
    end
end

%% 6B. PERSISTENCE MASK -- reject whatever is always there
% -------------------------------------------------------------------------

clutterBin = false(numAdcSamples, 1);
if ~isempty(params.persistFrac)
    occ = sum(detections, 2) / numFrames;
    clutterBin = occ > params.persistFrac;
    for d = 1:params.persistDilate       % include the skirt, not just the peak
        clutterBin = clutterBin | [false; clutterBin(1:end-1)] ...
                                | [clutterBin(2:end); false];
    end
    detections(clutterBin, :) = false;
    if any(clutterBin)
        fprintf('persistence  : masked %d bin(s) present in >%.0f%% of frames:', ...
                nnz(clutterBin), 100*params.persistFrac);
        fprintf(' %.2f m', rangeAxis(clutterBin));  fprintf('\n');
    end
    % A capture short relative to the event would mask the target itself.
    % Say so rather than silently deleting the thing being measured.
    if nnz(clutterBin) > 0.2*(gateHi-gateLo+1)
        fprintf(['  ^^ that is a large share of the gate. If the capture is\n' ...
                 '     short relative to the flight, raise params.persistFrac.\n']);
    end
end

%% 7. RANGE HISTORY r(t)
% -------------------------------------------------------------------------

t_s    = (0:numFrames-1).' * framePeriod;
snrDB       = zeros(numAdcSamples, numFrames);
snrDB(detections) = 10*log10(residual(detections) ./ noiseEst(detections));

pathBin = zeros(numFrames, 1);
if params.track.enable
    maxStep = max(1, ceil(params.track.vmax * framePeriod / rangeRes));
    fprintf('tracking     : Viterbi, <= %d bin/frame (%.0f m/s at %.0f Hz)\n', ...
            maxStep, params.track.vmax, 1/framePeriod);
    S = zeros(numAdcSamples, numFrames);
    S(detections) = max(snrDB(detections), 0);     % undetected bins score 0,
    V = -inf(numAdcSamples, numFrames);            % so gaps cost but do not
    B = zeros(numAdcSamples, numFrames, 'uint32'); % break the path
    V(gateLo:gateHi, 1) = S(gateLo:gateHi, 1);
    for f = 2:numFrames
        [best, arg] = sliding_max(V(:, f-1), maxStep);
        V(gateLo:gateHi, f) = S(gateLo:gateHi, f) + best(gateLo:gateHi);
        B(gateLo:gateHi, f) = arg(gateLo:gateHi);
    end
    [~, bEnd] = max(V(:, numFrames));
    pathBin(numFrames) = bEnd;
    for f = numFrames:-1:2
        pathBin(f-1) = double(B(pathBin(f), f));
    end
else
    for f = 1:numFrames
        idx = find(detections(:, f));
        if isempty(idx); continue; end
        [~, k] = max(residual(idx, f));
        pathBin(f) = idx(k);
    end
end

rHat   = nan(numFrames, 1);
snrHat = nan(numFrames, 1);
angHat = nan(numFrames, 1);
for f = 1:numFrames
    r = pathBin(f);
    % Only report where the path actually sits on a detection. Bridging a gap
    % keeps the path alive, but the range history is not valid there.
    if r < 1 || ~detections(r, f); continue; end
    rHat(f) = rangeAxis(r);
    % Sub-bin parabolic interpolation. Without it r(t) is quantised to the
    % bin width and the fit inherits that quantisation as its error floor.
    if r > 1 && r < numAdcSamples
        y = log(residual(r-1:r+1, f) + eps);
        den = y(1) - 2*y(2) + y(3);
        if den ~= 0
            d = 0.5 * (y(1) - y(3)) / den;
            if abs(d) <= 1; rHat(f) = rangeAxis(r) + d * rangeRes; end
        end
    end
    snrHat(f) = snrDB(r, f);
    if nAng > 1
        wsp = 2*pi*(double(angIdx(r,f))-1)/nAng;
        wsp = mod(wsp + pi, 2*pi) - pi;             % wrap to (-pi, pi]
        angHat(f) = -asin(max(min(wsp/pi, 1), -1)) * 180/pi;
    end
end
valid = ~isnan(rHat);
fprintf('range history: %d of %d frames (%.1f %%), median SNR %.1f dB\n', ...
        nnz(valid), numFrames, 100*nnz(valid)/numFrames, median(snrHat(valid)));
if any(~isnan(angHat))
    fprintf('coarse angle : %+.1f deg median, IQR %.1f (sign checked in sim only)\n', ...
            median(angHat(valid), 'omitnan'), iqr(angHat(valid)));
end

%% 7B. CONTEXT FOR THE ANGLE ESTIMATOR
% -------------------------------------------------------------------------

actx = struct( ...
    'raw',             raw,                  'bytesPerFrame',  bytesPerFrame, ...
    'deinterleave',    params.deinterleave,  'doConj',         doConj, ...
    'wRange',          wRange,               'clutterC',       clutterC, ...
    'clutterMode',     params.clutter,       'numAdcSamples',  numAdcSamples, ...
    'numRx',           numRx,                'numTx',          numTx, ...
    'numLoops',        numLoops,             'numVirtual',     numVirtual, ...
    'samplesPerFrame', samplesPerFrame,      'numFrames',      numFrames, ...
    'rangeAxis',       rangeAxis,            'rHat',           rHat, ...
    'snrHat',          snrHat,               'valid',          valid, ...
    't_s',             t_s,                  'framePeriod',    framePeriod, ...
    'chirpPeriod',     chirpPeriod,          'wavelength',     wavelength);

%% 8. r^2(t) FIT
% -------------------------------------------------------------------------
% r^2(t) = |v|^2 t^2 + 2(p0.v)t + |p0|^2 for straight-line CONSTANT-velocity
% motion. 
fitMask = valid;
if ~isempty(params.fitWindow)
    fitMask = fitMask & t_s >= params.fitWindow(1) & t_s <= params.fitWindow(2);
end
speedEst = NaN; resRMS = NaN; pFit = [];
if nnz(fitMask) >= 10
    tf = t_s(fitMask);  rf = rHat(fitMask);
    % Fit in time measured from start of the fit window
    tOrg = tf(1);
    for iter = 1:5                          % robust reweighting
        pFit  = polyfit(tf - tOrg, rf.^2, params.fitDeg);
        res   = rf - sqrt(max(polyval(pFit, tf - tOrg), 0));
        s     = 1.4826 * median(abs(res - median(res)));
        if s <= 0; break; end
        keep = abs(res - median(res)) < 3*s;
        if all(keep) || nnz(keep) < 10; break; end
        tf = tf(keep); rf = rf(keep);
    end
    resRMS = sqrt(mean((rf - sqrt(max(polyval(pFit, tf - tOrg), 0))).^2));

    if pFit(end-2) > 0; speedEst = sqrt(pFit(end-2)); end
    fprintf('--- r^2(t) fit: %.2f s, %d points, order %d ---\n', ...
            tf(end)-tf(1), numel(tf), params.fitDeg);

    fprintf('  |v| at fit start   : %.2f m/s\n', speedEst);
    fprintf('  range at fit start : %.2f m   (= |p0| iff window opens at launch)\n', ...
            sqrt(max(pFit(end),0)));
    fprintf('  residual RMS       : %.1f cm  (range bin %.1f cm)\n', ...
            resRMS*100, rangeRes*100);
    if resRMS > rangeRes
        fprintf(['  ^^ residual EXCEEDS one range bin. A degree-%d polynomial in\n' ...
                 '     r^2(t) does not describe this motion. Fix the test, not the fit.\n'], ...
                 params.fitDeg);
    end
else
    fprintf('--- r^2(t) fit skipped: only %d usable points ---\n', nnz(fitMask));
end

%% 8A. FLIGHT-MODEL FIT  (drag + gravity, same model as forward propagation)
% -------------------------------------------------------------------------

flightFit = [];  angFit = [];
if params.flight.enable && ~isnan(speedEst) && exist('tf','var') && numel(tf) >= 8
    % Tilt enters fit_flight through gravity, so a wrong one tips the whole
    % launch angle. Substituting 0 would just be a quieter wrong answer.
    assert(~isnan(params.flight.tiltDeg), ['params.flight.tiltDeg is NaN.' NL ...
        'Measure the radar tilt on the rig and set it, UP-POSITIVE: a radar' NL ...
        'mounted above the ball tilts DOWN, so the value is negative.']);

    [tok, idxf] = ismember(tf, t_s);
    assert(all(tok), ['section 8A: fit times not found in t_s. tf is built by\n' ...
        'logical indexing off t_s with no arithmetic, so this should be exact.\n' ...
        'If it is not, section 8 has changed and this mapping must change too.']);

    % Real azimuth AND elevation from the virtual array, on the fit frames.
    % The coarse angHat from integrate_cube is a 4-element by-product and a
    % consistency check only
    azf = nan(size(tf));  elf = nan(size(tf));
    aOpt = struct('snrMin', params.flight.angSnrMin, 'frames', idxf, ...
                  'unwrapElev', params.flight.unwrapElev, 'verbose', true);
    angFit = angle_history(actx, aOpt);
    [tok2, pos] = ismember(idxf, angFit.frames);
    azf(tok2) = angFit.azDeg(pos(tok2));
    elf(tok2) = angFit.elDeg(pos(tok2));


    if abs(angFit.diag.corrAzEl) > 0.5
        fprintf(['  A3 FAILED (r = %+.3f): dropping elevation from the fit.\n' ...
                 '  Resolve the Tx geometry before trusting a launch angle.\n'], ...
                angFit.diag.corrAzEl);
        elf(:) = NaN;
    end

    % Weight the angle residuals by the MEASURED frame-to-frame scatter
    sigAzUse = max(params.flight.sigAzDeg, angFit.sdAz);
    sigElUse = max(params.flight.sigElDeg, angFit.sdEl);
    fprintf('  angle sigmas used  : az %.2f deg, el %.2f deg (measured sd %.2f / %.2f)\n', ...
            sigAzUse, sigElUse, angFit.sdAz, angFit.sdEl);

    fOpt = struct('p0',         params.flight.p0, ...
                  'k',          params.flight.k, ...
                  'ballType',   params.flight.ballType, ...
                  'tiltDeg',    params.flight.tiltDeg, ...
                  'fixElevDeg', params.flight.fixElevDeg, ...
                  'sigR',       params.flight.sigR, ...
                  'sigAzDeg',   sigAzUse, ...
                  'sigElDeg',   sigElUse, ...
                  'seed',       [speedEst, median(elf,'omitnan'), ...
                                 median(azf,'omitnan'), tf(1)]);
    if isnan(fOpt.seed(2)), fOpt.seed(2) = 10; end
    if isnan(fOpt.seed(3)), fOpt.seed(3) = 0;  end
    % If no usable elevation discard and report to user
    if all(isnan(elf)) && isempty(params.flight.fixElevDeg)
        fOpt.fixElevDeg = fOpt.seed(2);
        fprintf('  no usable elevation: holding it at %+.1f deg.\n', fOpt.fixElevDeg);
    end
    flightFit = fit_flight(tf, rf, azf, elf, fOpt);

    fprintf('  polynomial (deg %d) : %.3f m/s\n', params.fitDeg, speedEst);
    fprintf('  flight model        : %.3f m/s   (%+.2f %% vs polynomial)\n', ...
            flightFit.speed, 100*(flightFit.speed - speedEst)/speedEst);
    fprintf(['  The polynomial is expected to read LOW: drag pulls the t^2\n' ...
             '  coefficient down by roughly k*|p0|/2. A difference in the\n' ...
             '  other direction means something else is wrong.\n']);
elseif params.flight.enable
    fprintf('--- flight-model fit skipped: no usable r^2 fit to seed from ---\n');
end

%% 8C. FORWARD PROPAGATION -- landing point and uncertainty band
% -------------------------------------------------------------------------

prop = [];
if params.propagate.enable && ~isempty(flightFit)
    % Same treatment as tiltDeg above -> the launch height scales the carry
    assert(~isnan(params.propagate.radarHeight), ...
        ['params.propagate.radarHeight is NaN.' NL ...
         'Measure the radar phase centre above the GROUND and set it. It' NL ...
         'sets the launch height and therefore the carry directly.']);

    pOpt = struct('radarHeight', params.propagate.radarHeight, ...
                  'netRange',    params.propagate.netRange, ...
                  'kFrac',       params.propagate.kFrac, ...
                  'nDraws',      params.propagate.nDraws, ...
                  'verbose',     true);
    prop = propagate_flight(flightFit, pOpt);
elseif params.propagate.enable
    fprintf(['--- forward propagation skipped: no flight-model fit. It needs\n' ...
             '    that fit''s covariance, which the polynomial fit cannot supply. ---\n']);
end

%% 8D. SPIN FROM MICRO-DOPPLER
% -------------------------------------------------------------------------
% Independent of the launch-state chain above
% it reads the same range bin but measures the WIDTH of the Doppler band 
% rather than its position. 
% Runs off actx, so it sees the same clutter cancellation as the angle
% estimator. 
spin = [];
if params.spin.enable
    if numTx > 1
        fprintf(['--- spin: %d transmitters, so v_max is a third of the\n' ...
                 '    single-Tx figure and the band will fold. Capture spin\n' ...
                 '    on awr1843_spin_16l.json, on its own shot. ---\n'], numTx);
    end
    sOpt = struct('shape',    params.spin.shape, ...
                  'sinAlpha', params.spin.sinAlpha, ...
                  'snrMin',   params.spin.snrMin, ...
                  'ballType', params.flight.ballType, ...
                  'verbose',  true);
    spin = spin_history(actx, sOpt);
end

%% 8B. DIAGNOSTIC FRAME SELECTION
% -------------------------------------------------------------------------
%
%   dFrame  best-SNR detected frame inside the fit window. Answers "did the
%           target survive clutter cancellation, and where does it sit in
%           Doppler" -- including whether it has folded, which is the
%           evidence for the deliberate ambiguity trade.
%   qFrame  a representative target-free frame: the residual clutter floor
%           the target has to beat. Median rather than minimum, so it is a
%           typical floor and not a lucky one.
%
skipN    = min(params.diagSkipFrames, numFrames - 1);
eligible = false(numFrames, 1);  eligible(skipN+1:end) = true;
gatePeak = max(residual(gateLo:gateHi, :), [], 1).';

if ~isempty(params.fitWindow)
    evt  = t_s >= params.fitWindow(1) & t_s <= params.fitWindow(2);
    pad  = 0.5 * diff(params.fitWindow);          % half a window each side
    evtP = t_s >= params.fitWindow(1) - pad & t_s <= params.fitWindow(2) + pad;
else
    evt  = valid;                                  % no window: the detections
    evtP = evt;                                    % are all we know about
end

if ~isempty(params.diagFrame)
    dFrame = params.diagFrame;
else
    cand = fitMask & eligible & ~isnan(snrHat);
    if ~any(cand); cand = valid & eligible;  end      % no fit -- any detection
    if ~any(cand); cand = eligible;          end      % no detections at all
    score = -inf(numFrames, 1);
    score(cand) = snrHat(cand);
    if all(~isfinite(score(cand)))
        score(cand) = gatePeak(cand);                 % fall back to raw power
    end
    [~, dFrame] = max(score);
end

if ~isempty(params.diagQuietFrame)
    qFrame = params.diagQuietFrame;
else
   % Find a frame that is eligible and does not contain the target
    quiet = eligible & ~evtP & ~valid;            % clear of the event, nothing seen
    if ~any(quiet); quiet = eligible & ~evtP;  end   % clear of the event
    if ~any(quiet); quiet = eligible & ~evt;   end   % outside the fit window
    if ~any(quiet); quiet = eligible & ~valid; end   % nothing seen
    if ~any(quiet); quiet = eligible;          end
    qIdx = find(quiet);
    [~, k] = min(abs(gatePeak(qIdx) - median(gatePeak(qIdx))));
    qFrame = qIdx(k);
end

fprintf('diag frames  : target %d (t %.3f s, SNR %.1f dB), quiet %d (t %.3f s)\n', ...
        dFrame, t_s(dFrame), snrHat(dFrame), qFrame, t_s(qFrame));
if evtP(qFrame)
    fprintf(['  ^^ no frame outside the event survived: the quiet frame is NOT\n' ...
             '     target-free. Read the grey trace with that in mind.\n']);
end
if ~valid(dFrame)
    fprintf(['  ^^ no CFAR detection on the diagnostic frame: nothing was found\n' ...
             '     anywhere in the fit window, so the panels below show clutter.\n']);
end

% Range-Doppler for dFrame, recomputed here rather than cached during pass 2:
% which frame is worth looking at is not known until the whole capture has
% been processed
rpD = frame_rp(raw, dFrame, bytesPerFrame, params.deinterleave, doConj, ...
    numAdcSamples, numRx, numTx, numLoops, numVirtual, samplesPerFrame, wRange);
if useClutter; rpD = rpD - clutterC; end
if params.rdRemoveStatic;              rpD = rpD - mean(rpD, 3); end
wDop = get_window(params.windowType, numLoops);
wDop = reshape(wDop / sqrt(mean(wDop.^2)), [1 1 numLoops]);
nfft = max(params.dopplerNFFT, numLoops);
rdDiag = squeeze(sum(abs(fftshift(fft(rpD .* wDop, nfft, 3), 3)).^2, 2));
dopplerAxisDisp = (-nfft/2 : nfft/2-1).' * (2*maxVelocity / nfft);

%% 9. VISUALISATION
% -------------------------------------------------------------------------
figure('Name', 'FMCW pipeline', 'Color', 'w', 'Position', [60 60 1500 820]);
inGate = rangeAxis >= params.rangeGate(1) & rangeAxis <= params.rangeGate(2);

ax1 = subplot(2,3,[1 2]);
rtdB = 10*log10(residual + eps);
imagesc(t_s, rangeAxis, rtdB); axis xy; colormap(ax1, jet); colorbar;
cmax = max(max(rtdB(inGate,:))); caxis([cmax-45, cmax]);
ylim(params.rangeGate);
hold on;
plot(t_s(valid), rHat(valid), '.', 'Color', 'w', 'MarkerSize', 7);
if any(clutterBin)      % what the persistence mask threw away, and why
    for b = find(clutterBin).'
        plot(t_s([1 end]), rangeAxis(b)*[1 1], '--', 'Color', [1 1 1 0.55]);
    end
end
hold off;
title('Range-time, clutter cancelled (dB); dashed = persistence-masked');
xlabel('t (s)'); ylabel('Range (m)');
light_axes(ax1);

ax2 = subplot(2,3,3);
plot(t_s(valid), rHat(valid), '.', 'Color', [0 0.35 0.85], 'MarkerSize', 9); hold on;
if ~isempty(pFit)
    tt = linspace(min(t_s(fitMask)), max(t_s(fitMask)), 200).';
    plot(tt, sqrt(max(polyval(pFit, tt), 0)), '-', ...
         'Color', [0.85 0.1 0.1], 'LineWidth', 1.8);
    legend('r(t)', 'r^2(t) fit', 'Location', 'best');
end
hold off; grid on; ylim(params.rangeGate);
title('Range history r(t)'); xlabel('t (s)'); ylabel('r (m)');
light_axes(ax2);

ax3 = subplot(2,3,4);
% Three traces, three separate questions: did the statics come down (blue vs
% red), does the target stand above the residual floor (red vs grey), and is
% that floor flat across the gate (grey on its own).
plot(rangeAxis, 10*log10(rawTime(:,dFrame)+eps), '-', ...
     'Color', [0 0.35 0.85], 'LineWidth', 1.4); hold on;
plot(rangeAxis, 10*log10(residual(:,dFrame)+eps), '-', ...
     'Color', [0.85 0.1 0.1], 'LineWidth', 1.4);
plot(rangeAxis, 10*log10(residual(:,qFrame)+eps), '-', ...
     'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
grid on; xlabel('Range (m)'); ylabel('Power (dB)');
yl = ylim; plot(params.rangeGate([1 1]), yl, 'k--');
plot(params.rangeGate([2 2]), yl, 'k--'); hold off;
title(sprintf('Range profile, frame %d (t = %.2f s)', dFrame, t_s(dFrame)));
legend(sprintf('frame %d, before cancel', dFrame), ...
       sprintf('frame %d, after', dFrame), ...
       sprintf('frame %d, after (target-free)', qFrame), 'Location', 'best');
light_axes(ax3);

ax4 = subplot(2,3,5);
if ~isempty(rdDiag)
    rddB = 10*log10(rdDiag + eps);
    imagesc(dopplerAxisDisp, rangeAxis, rddB); axis xy; colormap(ax4, jet); colorbar;
    ylim(params.rangeGate);
    cm = max(max(rddB(inGate,:))); caxis([cm-40, cm]);
    if ~isnan(rHat(dFrame))    % where the range CFAR put it, for cross-check
        hold on;
        plot(dopplerAxisDisp([1 end]), rHat(dFrame)*[1 1], 'w--', 'LineWidth', 1.2);
        hold off;
    end
    title(sprintf('Range-Doppler, frame %d (t = %.2f s)', dFrame, t_s(dFrame)));
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
end
light_axes(ax4);

ax5 = subplot(2,3,6);
plot(t_s(valid), snrHat(valid), '.', 'Color', [0 0.5 0.2], 'MarkerSize', 9);
grid on; title('Peak SNR over clutter'); xlabel('t (s)'); ylabel('dB');
light_axes(ax5);

%% HELPER FUNCTIONS
% -------------------------------------------------------------------------
function [p, iAng] = integrate_cube(rp, mode, nAng, nDop, nRx, nTx, nL)
% Collapse [range, virtual, loop] to one power value per range bin.
%
% 'noncoherent': sum |x|^2 over every channel and loop. Simple, robust, and
%   what this pipeline did originally -- but for N channels it improves SNR
%   by only ~sqrt(N), because it throws the phase relationships away before
%   adding.
%
% 'coherent': FFT across the 4 Rx (an angle transform -- their lambda/2
%   spacing is verified in mimo_angle.m section A1) and across the loops (a
%   Doppler transform), then take the strongest cell. A single dominant
%   scatterer concentrates into one angle-Doppler cell and gains ~N, while
%   the noise gains only the max-of-N inflation.
%
    nS = size(rp, 1);
    if strcmpi(mode, 'noncoherent')
        p = sum(sum(abs(rp).^2, 3), 2);
        iAng = ones(nS, 1, 'uint16');
        return;
    end
    Z = reshape(rp, [nS, nRx, nTx, nL]);
    Z = fft(Z, nAng, 2);                        % angle, across Rx
    Z = fft(Z, nDop, 4);                        % doppler, across loops
    Q = reshape(sum(abs(Z).^2, 3), [nS, nAng*nDop]);   % Tx summed in power
    [p, k] = max(Q, [], 2);
    iAng = uint16(mod(double(k)-1, nAng) + 1);  % angle index of the winner
end

function [m, a] = sliding_max(v, w)
% Max of v over a +/-w window, with the argmax. w is a few bins, so looping
% over the shifts beats anything cleverer and stays obviously correct.
    n = numel(v);
    m = -inf(n, 1);
    a = uint32((1:n).');            % default: point at self, never index 0
    for d = -w:w
        idx  = (1:n).' + d;
        ok   = idx >= 1 & idx <= n;
        cand = -inf(n, 1);
        cand(ok) = v(idx(ok));
        better = cand > m;
        m(better) = cand(better);
        a(better) = uint32(idx(better));
    end
end

function light_axes(ax)
% Force light axes regardless of the MATLAB desktop theme. Under the dark
% theme the axes background goes near-black and dark markers vanish, which
% reads as "no data" rather than "wrong colour".
    set(ax, 'Color', 'w', 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], ...
            'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.9, 'Box', 'on');
    set(get(ax,'Title'),  'Color', 'k');
    set(get(ax,'XLabel'), 'Color', 'k');
    set(get(ax,'YLabel'), 'Color', 'k');
end

function bgC = bg_mean_profile(bgDir, ref, params, doConj, w, nV, nExp, tgtMeta)
%BG_MEAN_PROFILE  Mean complex range profile of a target-free background capture.
%
%  Range-processed through exactly the same chain as the target capture: any
%  difference in window, I/Q order or deinterleave would survive the
%  subtraction and read as clutter.
%
%  The dimension checks are hard errors, not warnings. Two captures on
%  different profiles subtract to something numerically fine and physically
%  meaningless, which is precisely the silent-failure mode this codebase is
%  already prone to (CLAUDE.md section 6).
    NL       = char(10);  %#ok<CHARTEN>
    metaFile = fullfile(bgDir, 'meta.json');
    binFile  = fullfile(bgDir, 'frames.bin');
    assert(isfile(metaFile), 'background: meta.json not found in %s', bgDir);
    assert(isfile(binFile),  'background: frames.bin not found in %s', bgDir);

    bgMeta = jsondecode(fileread(metaFile));
    d      = bgMeta.dims;

    flds = {'n_samples', 'n_rx', 'n_tx', 'n_loops', 'frame_bytes'};
    for i = 1:numel(flds)
        got = double(d.(flds{i}));
        assert(got == ref.(flds{i}), ...
            ['background capture has %s = %d, target has %d.' NL ...
             'Different profile -- the two are not comparable. Re-record the' NL ...
             'background on the same config without restarting the driver.'], ...
             flds{i}, got, ref.(flds{i}));
    end
    if isfield(bgMeta, 'config_path') && isfield(tgtMeta, 'config_path') && ...
            ~strcmp(bgMeta.config_path, tgtMeta.config_path)
        fprintf(['  ^^ background profile is %s but target is %s. Dimensions\n' ...
                 '     match and the profiles do not -- check this is deliberate.\n'], ...
                 bgMeta.config_path, tgtMeta.config_path);
    end

    bpf  = double(d.frame_bytes);
    info = dir(binFile);
    assert(mod(info.bytes, bpf) == 0, ...
        'background frames.bin is %d B, not a whole multiple of %d B/frame.', ...
        info.bytes, bpf);
    nF = info.bytes / bpf;
    assert(nF >= 20, ['background capture has only %d frames. Its mean IS the' NL ...
        'clutter estimate, so too few frames means a noisy subtrahend that ADDS' NL ...
        'noise to every frame of the target capture.'], nF);

    fid   = fopen(binFile, 'r');
    rawBg = fread(fid, Inf, 'int16=>int16');
    fclose(fid);

    nS  = double(d.n_samples);  nRx = double(d.n_rx);
    nTx = double(d.n_tx);       nL  = double(d.n_loops);

    bgC = complex(zeros(nS, nV, nL));
    for f = 1:nF
        bgC = bgC + frame_rp(rawBg, f, bpf, params.deinterleave, doConj, ...
                             nS, nRx, nTx, nL, nV, nExp, w);
    end
    bgC = bgC / nF;
    fprintf('background   : %s\n', bgDir);
    fprintf('  frames     : %d averaged\n', nF);
end

function rp = frame_rp(raw, f, bytesPerFrame, mode, doConj, nS, nRx, nTx, nL, nV, nExp, w)
% One frame: bytes -> complex -> radar cube -> windowed range FFT.
    i0    = (f-1)*bytesPerFrame/2;
    chunk = double(raw(i0 + (1:bytesPerFrame/2)));
    cplx  = deinterleave(chunk, mode);
    if doConj; cplx = conj(cplx); end
    assert(numel(cplx) == nExp, 'frame %d: %d complex samples, expected %d', ...
           f, numel(cplx), nExp);
    cube = reshape(cplx, [nS, nRx, nTx, nL]);
    cube = reshape(cube, [nS, nV, nL]);
    rp   = fft(cube .* w, nS, 1);
end

function cplx = deinterleave(chunk, mode)
% AWR1843 streams complex data over 2 LVDS lanes as groups of four int16:
% I0 I1 Q0 Q1 -- not alternating I,Q.
    switch lower(mode)
        case 'lvds4'
            assert(mod(numel(chunk), 4) == 0, 'lvds4 needs a multiple of 4 int16');
            gq = reshape(chunk, 4, []);
            cplx = reshape([gq(1,:) + 1j*gq(3,:); gq(2,:) + 1j*gq(4,:)], [], 1);
        case 'iq2'
            cplx = chunk(1:2:end) + 1j*chunk(2:2:end);
        otherwise
            error('unknown de-interleave mode "%s"', mode);
    end
end

function win = get_window(winType, N)
% Computed directly rather than via hann()/hamming()/blackman(), which need
% the Signal Processing Toolbox. 
    if N == 1; win = 1; return; end
    n = (0:N-1).' / (N-1);
    switch lower(winType)
        case 'hann',     win = 0.5 - 0.5*cos(2*pi*n);
        case 'hamming',  win = 0.54 - 0.46*cos(2*pi*n);
        case 'blackman', win = 0.42 - 0.5*cos(2*pi*n) + 0.08*cos(4*pi*n);
        case 'rect',     win = ones(N, 1);
        otherwise,       error('unknown window "%s"', winType);
    end
end
