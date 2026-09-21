%% =========================================================================
%  FMCW RADAR DETECTION & SIGNAL PROCESSING PIPELINE
%  =========================================================================
clear; clc; close %

%% 1. CONFIGURATION & TUNABLE PARAMETERS (EDIT ON THE FLY)
% -------------------------------------------------------------------------
% File Paths
configFileName = 'C:\Users\Chris Scheepers\Desktop\results\2026-08-29_14-12-44_wideband_01\meta.json'; % Save your JSON configuration here
binFileName    = 'C:\Users\Chris Scheepers\Desktop\results\2026-08-29_14-12-44_wideband_01\frames.bin';        % Raw DCA1000 binary data

% Signal Processing & Filtering Thresholds
params.enableMTI          = true;  % Clutter removal (zero-Doppler filter)
params.windowType         = 'hann';% Windowing: 'hann', 'hamming', 'blackman', 'rect'
params.targetFrameIdx     = 1;     % Frame number to analyze

% 2D CA-CFAR Thresholds (Range-Doppler)
params.cfar.guardRange    = 2;     % Number of guard cells (Range)
params.cfar.guardDoppler  = 2;     % Number of guard cells (Doppler)
params.cfar.trainRange    = 4;     % Number of training cells (Range)
params.cfar.trainDoppler  = 4;     % Number of training cells (Doppler)
params.cfar.thresholdOffset = 10;  % CFAR Threshold offset (in dB)

% Spatial & Clustering (DBSCAN) Parameters
params.dbscan.epsilon     = 1.5;   % Neighborhood search radius (meters/bins)
params.dbscan.minPts      = 2;     % Minimum core points per cluster

%% 2. DYNAMIC JSON PARSING & KINEMATIC RESOLUTION SETUP
% -------------------------------------------------------------------------
cfg = jsondecode(fileread(configFileName));

% Constants
c = 3e8; % Speed of light (m/s)

% Dynamic Channel Mask Extraction
rxMask = cfg.channelCfg.rxChannelEn;
txMask = cfg.channelCfg.txChannelEn;
numRx  = bitcount_helper(rxMask); % e.g., 15 (0b1111) -> 4 RX
numTx  = bitcount_helper(txMask); % e.g., 7  (0b0111) -> 3 TX
numVirtualAntennas = numTx * numRx;

% Profile & Chirp Kinematics
numAdcSamples  = cfg.profileCfg.numAdcSamples;       % e.g., 448
sampleRate     = cfg.profileCfg.digOutSampleRate * 1e3; % e.g., 12500 ksps -> 12.5 MHz[cite: 1]
freqSlope      = cfg.profileCfg.freqSlopeConst * 1e12;  % e.g., 100 MHz/us -> 1e14 Hz/s[cite: 1]
startFreq      = cfg.profileCfg.startFreq * 1e9;        % e.g., 77 GHz[cite: 1]
wavelength     = c / startFreq;

% Frame & Loop Configuration
chirpStartIdx  = cfg.frameCfg.chirpStartIndex; %[cite: 1]
chirpEndIdx    = cfg.frameCfg.chirpEndIndex; %[cite: 1]
numTxPerLoop   = (chirpEndIdx - chirpStartIdx + 1); %[cite: 1]
numLoops       = cfg.frameCfg.numChirps;      % e.g., 4 loops per frame[cite: 1]
totalChirpsPerFrame = numTxPerLoop * numLoops;

% PRF & Resolutions
idleTime       = cfg.profileCfg.idleTime * 1e-6; %[cite: 1]
rampEndTime    = cfg.profileCfg.rampEndTime * 1e-6; %[cite: 1]
chirpPeriod    = idleTime + rampEndTime; % Tc
framePeriod    = cfg.frameCfg.framePeriod * 1e-3; % e.g., 6 ms[cite: 1]

% Derived Radar Metrics
bandwidth      = freqSlope * (numAdcSamples / sampleRate);
rangeRes       = c / (2 * bandwidth);
maxRange       = (sampleRate * c) / (2 * freqSlope);
velocityRes    = wavelength / (2 * numLoops * (numTxPerLoop * chirpPeriod));
maxVelocity    = wavelength / (4 * (numTxPerLoop * chirpPeriod));

fprintf('================ RADAR CONFIGURATION ================\n');
fprintf('Active Channels     : %d TX, %d RX -> %d Virtual Channels\n', numTx, numRx, numVirtualAntennas);
fprintf('Samples / Chirps    : %d Samples, %d Chirps/Frame\n', numAdcSamples, totalChirpsPerFrame);
fprintf('Range Resolution    : %.3f m (Max Range: %.2f m)\n', rangeRes, maxRange);
fprintf('Velocity Resolution : %.3f m/s (Max Velocity: ±%.2f m/s)\n', velocityRes, maxVelocity);
fprintf('=====================================================\n');

%% 3. PARSE RAW BINARY DATA (DCA1000 FORMAT)
% -------------------------------------------------------------------------
fid = fopen(binFileName, 'r');
if fid == -1
    error('Binary data file "%s" could not be opened.', binFileName);
end
rawBytes = fread(fid, 'int16');
fclose(fid);

% Reshape bytes: 4 RX channels * 2 (I and Q) = 8 rows
adcData = reshape(rawBytes, numRx * 2, []);

% Separate I (top half) and Q (bottom half)
I = adcData(1:numRx, :);
Q = adcData(numRx+1:numRx*2, :);
complexData = I + 1j * Q; % Size: [numRx, total ADC samples across frame]

% Determine total frames and extract the target frame
samplesPerFrame = numAdcSamples * totalChirpsPerFrame;
totalFrames = floor(size(complexData, 2) / samplesPerFrame);

if params.targetFrameIdx > totalFrames
    error('Selected frame index exceeds available frames.');
end

frameOffset = (params.targetFrameIdx - 1) * samplesPerFrame;
frameData = complexData(:, frameOffset + 1 : frameOffset + samplesPerFrame);

% Reshape to: [RX, Fast-Time (Samples), Slow-Time (TX), Slow-Time (Loops)]
radarCubeRaw = reshape(frameData, [numRx, numAdcSamples, numTxPerLoop, numLoops]);

% Permute to Standard Format: [Range, Virtual Antennas, Doppler]
radarCube = permute(radarCubeRaw, [2, 1, 3, 4]); 
radarCube = reshape(radarCube, [numAdcSamples, numVirtualAntennas, numLoops]);

%% 4. RANGE & CLUTTER FILTERING (1D-FFT & MTI)
% -------------------------------------------------------------------------
% Apply windowing across fast-time
w_range = get_window(params.windowType, numAdcSamples);
rangeWinData = radarCube .* reshape(w_range, [numAdcSamples, 1, 1]);

% Range FFT
rangeProfile = fft(rangeWinData, numAdcSamples, 1);

% Clutter Suppression (MTI) across Slow-Time dimension
if params.enableMTI
    % Subtract average static background across chirps
    rangeProfile = rangeProfile - mean(rangeProfile, 3);
end

%% 5. DOPPLER PROCESSING (2D-FFT / RANGE-DOPPLER MAP)
% -------------------------------------------------------------------------
% Doppler windowing
w_doppler = get_window(params.windowType, numLoops);
dopplerWinData = rangeProfile .* reshape(w_doppler, [1, 1, numLoops]);

% Doppler FFT & shift zero-Doppler to center
rdMap = fftshift(fft(dopplerWinData, numLoops, 3), 3);

% Coherent/Non-coherent integration across virtual antennas for detection
rdSpectrum = squeeze(sum(abs(rdMap).^2, 2)); % [numAdcSamples x numLoops]
rdSpectrum_dB = 10 * log10(rdSpectrum + eps);

%% 6. 2D CA-CFAR DETECTION
% -------------------------------------------------------------------------
detectionMap = zeros(size(rdSpectrum_dB));
[numRangeBins, numDopplerBins] = size(rdSpectrum_dB);

gr = params.cfar.guardRange;
gd = params.cfar.guardDoppler;
tr = params.cfar.trainRange;
td = params.cfar.trainDoppler;
offset = params.cfar.thresholdOffset;

for r = (tr + gr + 1) : (numRangeBins - tr - gr)
    for d = (td + gd + 1) : (numDopplerBins - td - gd)
        % Extract training window
        fullWin = rdSpectrum_dB(r - (tr+gr) : r + (tr+gr), d - (td+gd) : d + (td+gd));
        
        % Mask out CUT and Guard cells
        cutAndGuard = rdSpectrum_dB(r - gr : r + gr, d - gd : d + gd);
        numTrainCells = numel(fullWin) - numel(cutAndGuard);
        
        noiseLevel = (sum(fullWin(:)) - sum(cutAndGuard(:))) / numTrainCells;
        
        % Threshold test
        if rdSpectrum_dB(r, d) > (noiseLevel + offset)
            detectionMap(r, d) = 1;
        end
    end
end

%% 7. POINT CLOUD EXTRACTION & SPATIAL CLUSTERING (DBSCAN)
% -------------------------------------------------------------------------
[detRangeIdx, detDopplerIdx] = find(detectionMap == 1);

% Convert indices to physical units
rangeAxis   = linspace(0, maxRange, numAdcSamples);
dopplerAxis = linspace(-maxVelocity, maxVelocity, numLoops);

detectedRanges    = rangeAxis(detRangeIdx)';
detectedVelocities = dopplerAxis(detDopplerIdx)';

if ~isempty(detectedRanges)
    % Cluster detections into cohesive target centroids
    features = [detectedRanges, detectedVelocities];
    clusterIdx = dbscan(features, params.dbscan.epsilon, params.dbscan.minPts);
else
    clusterIdx = [];
end

%% 8. VISUALIZATION
% -------------------------------------------------------------------------
figure('Name', 'FMCW Radar Sports Ball Detection', 'Color', 'w');

% Plot Range-Doppler Map
subplot(1, 2, 1);
imagesc(dopplerAxis, rangeAxis, rdSpectrum_dB);
axis xy; colormap(jet); colorbar;
title('Range-Doppler Spectrum (dB)');
xlabel('Velocity (m/s)'); ylabel('Range (m)');
hold on;
if ~isempty(detRangeIdx)
    plot(detectedVelocities, detectedRanges, 'r+', 'LineWidth', 1.5, 'MarkerSize', 8);
    legend('CFAR Detections', 'Location', 'northeast');
end
hold off;

% Plot Clustered Target Positions
subplot(1, 2, 2);
if ~isempty(clusterIdx)
    gscatter(detectedVelocities, detectedRanges, clusterIdx);
    title('DBSCAN Clustered Targets');
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
    grid on;
else
    title('No Targets Detected');
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
end

%% HELPER FUNCTIONS
% -------------------------------------------------------------------------
function count = bitcount_helper(val)
    count = sum(bitget(val, 1:32));
end

function win = get_window(winType, N)
    switch lower(winType)
        case 'hann',     win = hann(N);
        case 'hamming',  win = hamming(N);
        case 'blackman', win = blackman(N);
        case 'rect',     win = ones(N, 1);
        otherwise,       win = hann(N);
    end
end