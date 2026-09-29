%% =========================================================================
%  MIMO VIRTUAL ARRAY -- array checks and angle plots
%  =========================================================================
%  Run after fmcw_pipeline.m, in its workspace:   fmcw_pipeline; mimo_angle
%  Phase-comparison only (single dominant scatterer); no MUSIC/ESPRIT.
%  The estimator is in angle_history.m; 
%  This only checks the array and plots.
%  =========================================================================

assert(exist('actx', 'var') == 1, ...
   ['run fmcw_pipeline first, in this workspace:  fmcw_pipeline; mimo_angle' newline ...
    'This script reuses its detections, its raw ADC buffer and the actx' newline ...
    'context struct built in section 7B.']);

%% 0. PARAMETERS
% -------------------------------------------------------------------------
mimo.snrMin     = 15;     % dB, frames worth measuring an angle on
mimo.unwrapElev = false;  % extend elevation past +/- 30 deg by continuity

%% A/B. MEASURES THE ARRAY THEN THE ANGLES
% -------------------------------------------------------------------------
%     Current issues/open matters:
%     Rx phase slope linear and equal for all three Tx? (else channel mapping wrong)
%     Tx1->Tx3 phase step / Rx slope = Tx1-Tx3 separation in Rx spacings (expect 4).
%     Centred phase 2*phi(Tx2)-phi(Tx1)-phi(Tx3) independent of azimuth? (else Tx2 not
%     midway and elevation is contaminated; pipeline section 8A drops elevation on failure).
ang = angle_history(actx, struct('snrMin',     mimo.snrMin, ...
                                 'unwrapElev', mimo.unwrapElev, ...
                                 'verbose',    true));

sel   = ang.frames;
azDeg = ang.azDeg;
elDeg = ang.elDeg;

%% WHICH ANTENNA IS IN WHICH CHIRP SLOT
% -------------------------------------------------------------------------
% Slot -> antenna mapping sweep. 
% The mapping comes from the chirpCfg masks
if ~exist('mimo', 'var') || ~isfield(mimo, 'sweepTx') || mimo.sweepTx
    P = perms([1 2 3]);
    fprintf('\nA4  slot-to-antenna mapping sweep:\n');
    fprintf('     %-9s | %8s | %6s | %9s | %9s\n', ...
            'slots', 'D', 'n', 'sd elev', 'A3 corr');
    for ip = 1:size(P,1)
        try
            a = angle_history(actx, struct('snrMin', mimo.snrMin, ...
                                           'txOrder', P(ip,:), 'verbose', false));
            fprintf('     [%d %d %d]   | %+8.2f | %6d | %9.2f | %+9.3f\n', ...
                    P(ip,1), P(ip,2), P(ip,3), a.diag.D, a.diag.Dn, ...
                    a.sdEl, a.diag.corrAzEl);
        catch err
            fprintf('     [%d %d %d]   | failed: %s\n', P(ip,:), err.message);
        end
    end
end

%% C. PLOTS
% -------------------------------------------------------------------------
figure('Name', 'MIMO angle', 'Color', 'w', 'Position', [80 80 1300 700]);
ts = ang.t;

ax1 = subplot(2,2,1);
plot(ts, azDeg, '.', 'Color', [0 0.35 0.85], 'MarkerSize', 8);
grid on; xlabel('t (s)'); ylabel('Azimuth (deg)');
title(sprintf('Azimuth (sd %.2f deg/frame)', ang.sdAz)); light_axes_m(ax1);

ax2 = subplot(2,2,2);
plot(ts, elDeg, '.', 'Color', [0.85 0.1 0.1], 'MarkerSize', 8);
% Draw the +/-30 deg elevation wrap: a track that reaches it has folded.
yline(+ang.diag.elAmbigDeg, '--', '+30 deg wrap', 'Color', [0.5 0.5 0.5]);
yline(-ang.diag.elAmbigDeg, '--', '-30 deg wrap', 'Color', [0.5 0.5 0.5]);
grid on; xlabel('t (s)'); ylabel('Elevation (deg)');
title(sprintf('Elevation (sd %.2f deg/frame)', ang.sdEl)); light_axes_m(ax2);

ax3 = subplot(2,2,3);
plot(actx.rHat(sel), elDeg, '.', 'Color', [0.2 0.5 0.2], 'MarkerSize', 8);
grid on; xlabel('Range (m)'); ylabel('Elevation (deg)');
title('Elevation vs range'); light_axes_m(ax3);

ax4 = subplot(2,2,4);
plot(ang.u, ang.phiEl, '.', 'Color', [0.4 0.2 0.6], 'MarkerSize', 8);
grid on; xlabel('u = sin(az)cos(el)'); ylabel('centred phase (rad)');
title(sprintf('A3 leakage test, r = %+.3f', ang.diag.corrAzEl));
light_axes_m(ax4);

%% HELPERS
% -------------------------------------------------------------------------
function light_axes_m(ax)
    set(ax, 'Color', 'w', 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], ...
            'GridColor', [0.6 0.6 0.6], 'GridAlpha', 0.9, 'Box', 'on');
    set(get(ax,'Title'), 'Color', 'k');
    set(get(ax,'XLabel'), 'Color', 'k');
    set(get(ax,'YLabel'), 'Color', 'k');
end
