%% =========================================================================
%  CW PIPELINE:  raw ADC cube  ->  Doppler spectrum  ->  velocity history

clear; clc; close all;

%% 1. INPUT AND TUNABLE PARAMETERS
% -------------------------------------------------------------------------
sessionDir = 'C:\Users\Chris Scheepers\Desktop\results\2026-09-07_11-44-15_ultra_static';

% Display window for the velocity-time map. The unambiguous span is enormous
% and almost all of it is empty, so plotting it whole hides the signal.
params.velWindow  = [-80 80];    % m/s
% Ignore this band around zero when hunting the target tone: DC carries the
% Tx->Rx leakage and every static in the room.
params.dcGuard    = 3.0;         % m/s, +/- around zero

params.windowType = 'blackman';  % low sidelobes matter: a huge DC spike sits
                                 % right next to a small target tone
params.dcRemove   = true;        % subtract the fast-time mean per chirp
params.clutter    = true;        % also subtract the mean COMPLEX spectrum
                                 % across frames, as fmcw_pipeline.m does
params.nfftMult   = 2;           % zero-pad: interpolates, adds no resolution

% Bands used to measure the IF high-pass roll-off, as a fraction of Nyquist.
params.hpfLoBand  = [0.02 0.10]; % "below the corner"
params.hpfHiBand  = [0.60 0.95]; % "well above the corner"

%% 2. CONFIGURATION FROM THE CAPTURE ITSELF
% -------------------------------------------------------------------------
metaFile = fullfile(sessionDir, 'meta.json');
binFile  = fullfile(sessionDir, 'frames.bin');
assert(isfile(metaFile), 'meta.json not found in %s', sessionDir);
assert(isfile(binFile),  'frames.bin not found in %s', sessionDir);

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

sampleRate  = cfg.profileCfg.digOutSampleRate * 1e3;
freqSlope   = cfg.profileCfg.freqSlopeConst * 1e12;
startFreq   = cfg.profileCfg.startFreq * 1e9;
framePeriod = cfg.frameCfg.framePeriod * 1e-3;

%Check if FMCW or CW capture
assert(freqSlope == 0, ...
   ['freqSlopeConst = %g MHz/us -- this is an FMCW capture, not CW.\n' ...
    'Use fmcw_pipeline.m. This script assumes no ramp, so its frequency\n' ...
    'axis would be meaningless here.'], cfg.profileCfg.freqSlopeConst / 1e6);

wavelength = c / startFreq;              % no sweep: centre == start
dwell      = numAdcSamples / sampleRate;
velRes     = wavelength / (2 * dwell);
velMax     = wavelength * sampleRate / 4;


hpf1Tbl = [175e3 235e3 350e3 700e3];
hpf2Tbl = [350e3 700e3 1.4e6 2.8e6];
hpf1 = hpf1Tbl(cfg.profileCfg.hpfCornerFreq1 + 1);
hpf2 = hpf2Tbl(cfg.profileCfg.hpfCornerFreq2 + 1);
hpfBind = max(hpf1, hpf2);

fprintf('================ CW CAPTURE ================\n');
fprintf('session      : %s\n', meta.session);
fprintf('profile      : %s\n', meta.config_path);
fprintf('carrier      : %.1f GHz, lambda %.3f mm\n', startFreq/1e9, wavelength*1e3);
fprintf('dwell        : %.1f us (%d samples at %g ksps)\n', ...
        dwell*1e6, numAdcSamples, sampleRate/1e3);
fprintf('velocity     : res %.2f m/s, unambiguous +/-%.0f m/s\n', velRes, velMax);
fprintf('IF high-pass : %.0f kHz binding (hpf1 %.0f, hpf2 %.0f kHz)\n', ...
        hpfBind/1e3, hpf1/1e3, hpf2/1e3);
fprintf('  -> needs %.0f m/s to clear it; 70 m/s gives only %.1f kHz\n', ...
        wavelength*hpfBind/2, 2*70/wavelength/1e3);
fprintf('frame rate   : %.1f Hz\n', 1/framePeriod);

%% 3. READ THE RAW ADC STREAM
% -------------------------------------------------------------------------
info = dir(binFile);
bytesPerFrame = double(dims.frame_bytes);

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
    'frames.bin is %d B, not a multiple of %d B/frame.', info.bytes, bytesPerFrame);
numFrames = info.bytes / bytesPerFrame;
fprintf('frames.bin   : %d frames, %.2f s\n', numFrames, numFrames*framePeriod);

fid = fopen(binFile, 'r');
raw = fread(fid, Inf, 'int16=>int16');
fclose(fid);
samplesPerFrame = numAdcSamples * numRx * chirpsPerFrame;

doConj = (cfg.adcbufCfg.SampleSwap == 1);
fprintf('I/Q order    : SampleSwap=%d -> conjugate=%d\n', ...
        cfg.adcbufCfg.SampleSwap, doConj);
fprintf(['  NOTE: the SIGN of velocity follows this convention. Verify it\n' ...
         '  once against a target of known direction before trusting it.\n']);

wDop = get_window(params.windowType, numAdcSamples);
wDop = wDop / sqrt(mean(wDop.^2));
nfft = params.nfftMult * numAdcSamples;

freqAxis = (-nfft/2 : nfft/2-1).' * (sampleRate / nfft);
velAxis  = freqAxis * wavelength / 2;

%% 4. DOPPLER SPECTRUM PER FRAME
% -------------------------------------------------------------------------
% One spectrum per frame: FFT along fast time, power summed over the virtual
% antennas and the loops. Loops are repeated looks at the same instant of
% motion, so summing power averages the noise without smearing the tone.
specTime = zeros(nfft, numFrames);
specC    = complex(zeros(nfft, 1));      % coherent mean, for clutter removal

for f = 1:numFrames
    S = frame_spec(raw, f, bytesPerFrame, doConj, numAdcSamples, numRx, ...
        numTx, numLoops, numVirtual, samplesPerFrame, wDop, nfft, params.dcRemove);
    specTime(:, f) = sum(sum(abs(S).^2, 3), 2);
    specC = specC + S(:, 1, 1);
end
specC = specC / numFrames;

if params.clutter
    % Anything at a fixed beat frequency with fixed phase for the whole
    % capture is not a moving target
    residual = zeros(nfft, numFrames);
    for f = 1:numFrames
        S = frame_spec(raw, f, bytesPerFrame, doConj, numAdcSamples, numRx, ...
            numTx, numLoops, numVirtual, samplesPerFrame, wDop, nfft, params.dcRemove);
        S(:, 1, 1) = S(:, 1, 1) - specC;
        residual(:, f) = sum(sum(abs(S).^2, 3), 2);
    end
else
    residual = specTime;
end

%% 5. PEAK TONE PER FRAME
% -------------------------------------------------------------------------
inBand = abs(velAxis) > params.dcGuard;
vHat   = nan(numFrames, 1);
snrHat = nan(numFrames, 1);
for f = 1:numFrames
    col = residual(:, f);
    col(~inBand) = -inf;
    [pk, k] = max(col);
    if ~isfinite(pk); continue; end
    vHat(f) = velAxis(k);
    nl = median(residual(inBand, f));
    snrHat(f) = 10*log10(pk / max(nl, eps));
end
detected = snrHat > 10;
fprintf('tones > 10 dB: %d of %d frames (%.1f %%)\n', ...
        nnz(detected), numFrames, 100*nnz(detected)/numFrames);
if any(detected)
    fprintf('  peak velocity      : %+.2f m/s median, IQR %.2f (res %.2f)\n', ...
            median(vHat(detected)), iqr(vHat(detected)), velRes);
    fprintf('  tone SNR           : %.1f dB median, %.1f dB best\n', ...
            median(snrHat(detected)), max(snrHat(detected)));
end

%% 6. MEASURE THE IF HIGH-PASS FROM THE NOISE FLOOR
% -------------------------------------------------------------------------

meanSpec = mean(specTime, 2);            % pre-clutter-removal: want the floor
half     = freqAxis > 0;
fPos     = freqAxis(half);
sPos     = meanSpec(half);
nyq      = sampleRate / 2;

loSel = fPos > params.hpfLoBand(1)*nyq & fPos < params.hpfLoBand(2)*nyq;
hiSel = fPos > params.hpfHiBand(1)*nyq & fPos < params.hpfHiBand(2)*nyq;
pLo = median(sPos(loSel));
pHi = median(sPos(hiSel));
rolloff = 10*log10(pHi / pLo);

fprintf('--- IF high-pass, measured from the noise floor ---\n');
fprintf('  floor %.0f-%.0f kHz : %.1f dB\n', ...
        params.hpfLoBand(1)*nyq/1e3, params.hpfLoBand(2)*nyq/1e3, 10*log10(pLo));
fprintf('  floor %.0f-%.0f kHz : %.1f dB\n', ...
        params.hpfHiBand(1)*nyq/1e3, params.hpfHiBand(2)*nyq/1e3, 10*log10(pHi));
fprintf('  roll-off           : %.1f dB\n', rolloff);

% Fit a first-order high-pass |H(f)|^2 = A*f^2/(f^2+fc^2) to the floor.

fMeas = NaN;
if rolloff > 3
    fitSel = fPos > 0.01*nyq;
    ff = fPos(fitSel);  sdB = 10*log10(sPos(fitSel) + eps);
    fcTrial = logspace(log10(0.005*nyq), log10(0.95*nyq), 400);
    cost = inf(size(fcTrial));
    for k = 1:numel(fcTrial)
        gdB = 10*log10(ff.^2 ./ (ff.^2 + fcTrial(k)^2));
        cost(k) = mean((sdB - gdB - mean(sdB - gdB)).^2);   % gain profiled out
    end
    [~, kBest] = min(cost);
    if kBest > 1 && kBest < numel(fcTrial)      % reject a corner at the edge
        fMeas = fcTrial(kBest);
    end
end

if ~isnan(fMeas)
    fprintf('  measured corner    : %.0f kHz  (configured %.0f kHz)\n', ...
            fMeas/1e3, hpfBind/1e3);
    fprintf('  a 70 m/s ball sits %.1fx below it -> CW is NOT viable here\n', ...
            fMeas / (2*70/wavelength));
else
    fprintf('  measured corner    : none -- floor flat, roll-off only %.1f dB\n', ...
            rolloff);
    fprintf('  The IF filter is NOT blocking ball Doppler. CW is viable;\n');
    fprintf('  read the tone SNR above for how well.\n');
end

%% 7. VISUALISATION
% -------------------------------------------------------------------------
t_s = (0:numFrames-1).' * framePeriod;
figure('Name', 'CW pipeline', 'Color', 'w', 'Position', [60 60 1500 820]);

ax1 = subplot(2,2,[1 2]);
vtdB = 10*log10(residual + eps);
imagesc(t_s, velAxis, vtdB); axis xy; colormap(ax1, jet); colorbar;
ylim(params.velWindow);
inWin = velAxis >= params.velWindow(1) & velAxis <= params.velWindow(2);
cmax = max(max(vtdB(inWin,:))); caxis([cmax-45, cmax]);
hold on; plot(t_s(detected), vHat(detected), '.w', 'MarkerSize', 7); hold off;
title('Velocity-time (dB) -- CW, no range information');
xlabel('t (s)'); ylabel('Velocity (m/s)'); light_axes(ax1);

ax2 = subplot(2,2,3);
% THE key plot: the noise floor tracing the IF filter response.
plot(fPos/1e3, 10*log10(sPos + eps), '-', 'Color', [0 0.35 0.85], 'LineWidth', 1.3);
hold on; grid on; yl = ylim;
plot([hpf1 hpf1]/1e3, yl, '--', 'Color', [0.5 0.5 0.5]);
plot([hpf2 hpf2]/1e3, yl, 'k--', 'LineWidth', 1.3);
for v = [30 70]
    plot([1 1]*2*v/wavelength/1e3, yl, '-', 'Color', [0.85 0.1 0.1]);
end
if ~isnan(fMeas); plot([fMeas fMeas]/1e3, yl, 'g-', 'LineWidth', 1.5); end
hold off; xlabel('Beat frequency (kHz)'); ylabel('Mean power (dB)');
title('Noise floor vs IF high-pass'); set(gca, 'XScale', 'log');
legend({'measured floor', sprintf('hpf1 %.0f kHz', hpf1/1e3), ...
        sprintf('hpf2 %.0f kHz (binds)', hpf2/1e3), '30 m/s', '70 m/s'}, ...
        'Location', 'southeast');
light_axes(ax2);

ax3 = subplot(2,2,4);
yyaxis left;  plot(t_s(detected), vHat(detected), '.', 'MarkerSize', 9);
ylabel('Peak velocity (m/s)'); ylim(params.velWindow);
yyaxis right; plot(t_s, snrHat, '-', 'LineWidth', 0.8); ylabel('Tone SNR (dB)');
grid on; xlabel('t (s)'); title('Detected tone'); light_axes(ax3);

%% HELPER FUNCTIONS
% -------------------------------------------------------------------------
function light_axes(ax)
    set(ax, 'Color', 'w', 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], ...
            'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.9, 'Box', 'on');
    set(get(ax,'Title'), 'Color', 'k');
    set(get(ax,'XLabel'), 'Color', 'k');
    set(get(ax,'YLabel'), 'Color', 'k');
end

function S = frame_spec(raw, f, bytesPerFrame, doConj, nS, nRx, nTx, nL, nV, ...
                        nExp, w, nfft, dcRemove)
% One frame: bytes -> complex -> cube -> windowed DOPPLER FFT along fast time.
    i0    = (f-1)*bytesPerFrame/2;
    chunk = double(raw(i0 + (1:bytesPerFrame/2)));
    gq    = reshape(chunk, 4, []);          % LVDS 2-lane packing: I0 I1 Q0 Q1
    cplx  = reshape([gq(1,:) + 1j*gq(3,:); gq(2,:) + 1j*gq(4,:)], [], 1);
    if doConj; cplx = conj(cplx); end
    assert(numel(cplx) == nExp, 'frame %d: %d samples, expected %d', ...
           f, numel(cplx), nExp);
    cube = reshape(reshape(cplx, [nS, nRx, nTx, nL]), [nS, nV, nL]);
    if dcRemove
        % Statics beat at DC in CW. Remove the mean along fast time before
        % windowing, or the leakage spike's sidelobes bury everything.
        cube = cube - mean(cube, 1);
    end
    S = fftshift(fft(cube .* w, nfft, 1), 1);
end

function win = get_window(winType, N)
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
