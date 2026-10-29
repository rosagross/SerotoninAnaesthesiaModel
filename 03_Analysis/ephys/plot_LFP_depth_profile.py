
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (load_lfp, paths, remap, query_ephys_sessions, load_passive_opto_times,
                              init_one, figure_style)
from iblatlas.atlas import AllenAtlas
colors, dpi = figure_style()
ba = AllenAtlas()
one = init_one()
fig_path, save_path = paths()

# Settings
REGION = ['MOs']
YLIM = [-800, 500]
#REGION = ['VISa', 'VISpm']
#YLIM = [-500, 300]
UPPER_LAYER, LOWER_LAYER = '2/3', '5'  # depth is centered on the boundary between these
TIME_WIN = [1, 4]  # s before and after pulse onset
BASELINE = [-0.5, 0]  # baseline window (s) relative to pulse onset
FILTER_PAD = 1  # extra seconds loaded on both sides to avoid filter edge effects
LOWPASS = 40  # Hz, removes light artifacts
DOWNSAMPLE = 10  # downsample factor after low-pass filtering (2500 Hz -> 250 Hz)
DEPTH_GRID = np.arange(-990, 1000, 20)  # um relative to layer boundary, positive is superficial
CSD_SMOOTH = 1  # gaussian smoothing over depth (in grid bins) before computing the CSD
CONDUCTIVITY = 0.3  # S/m
PLOT_SMOOTH = 40  # um, sd of gaussian smoothing over depth of the averaged LFP for plotting
PLOT_SMOOTH_TIME = 0.02  # s, sd of gaussian smoothing over time of the averaged LFP for plotting

# Query sessions
rec = query_ephys_sessions(only_expression=True, one=one)

power_all, lfp_all, csd_all = [], [], []
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date}, {probe}')

    # Load in laser pulse times
    try:
        opto_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_times)} passive laser pulses')

    # Load in channels
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        channels = sl.load_channels()
    except Exception as err:
        print(err)
        continue
    if 'acronym' not in channels.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Get MOs channels and the layer 2/3 - 5 boundary
    beryl_regions = remap(channels['acronym'], dest='Beryl')
    region_channels = np.where(np.isin(beryl_regions, REGION))[0]
    # Only match the exact layer acronyms of the target region (e.g. MOs2/3 and MOs5)
    upper_chan = np.where(np.isin(channels['acronym'], [r + UPPER_LAYER for r in REGION]))[0]
    lower_chan = np.where(np.isin(channels['acronym'], [r + LOWER_LAYER for r in REGION]))[0]
    if (region_channels.size == 0) or (upper_chan.size == 0) or (lower_chan.size == 0):
        print(f'No regions found')
        continue
    boundary = (np.min(channels['axial_um'][upper_chan])
                + np.max(channels['axial_um'][lower_chan])) / 2
    rel_depths = channels['axial_um'][region_channels] - boundary

    # Load in LFP around every laser pulse
    lfp_trials = []
    for t, opto_time in enumerate(opto_times):
        lfp_arr, lfp_timestamps = load_lfp(
            eid, probe=probe, time_start=opto_time - TIME_WIN[0] - FILTER_PAD,
            time_end=opto_time + TIME_WIN[1] + FILTER_PAD, channels=region_channels, one=one)

        # Low-pass filter to get rid of light artifacts
        sampling_rate = 1 / np.median(np.diff(lfp_timestamps))
        sos = butter(4, LOWPASS, btype='lowpass', fs=sampling_rate, output='sos')
        lfp_arr = sosfiltfilt(sos, lfp_arr, axis=-1)

        # Cut off padding and downsample
        n_pad = int(np.round(FILTER_PAD * sampling_rate))
        n_samples = int(np.round((TIME_WIN[0] + TIME_WIN[1]) * sampling_rate))
        lfp_trials.append(lfp_arr[:, n_pad:n_pad + n_samples:DOWNSAMPLE])
    lfp_trials = np.stack(lfp_trials)  # trials x channels x time
    time_ax = (np.arange(0, n_samples, DOWNSAMPLE) / sampling_rate) - TIME_WIN[0]
    baseline_idx = (time_ax >= BASELINE[0]) & (time_ax < BASELINE[1])

    # Average channels at the same depth (side-by-side sites)
    unique_depths = np.unique(rel_depths)
    lfp_trials = np.stack([np.mean(lfp_trials[:, rel_depths == d, :], axis=1)
                           for d in unique_depths], axis=1)

    # LFP power: squared amplitude averaged over trials, as % change from baseline
    lfp_demean = lfp_trials - np.mean(lfp_trials, axis=2, keepdims=True)
    lfp_power = np.mean(lfp_demean ** 2, axis=0)
    baseline_power = np.mean(lfp_power[:, baseline_idx], axis=1, keepdims=True)
    lfp_power = ((lfp_power - baseline_power) / baseline_power) * 100

    # Evoked LFP: average over trials, baseline subtracted
    lfp_mean = np.mean(lfp_trials, axis=0)
    lfp_mean = lfp_mean - np.mean(lfp_mean[:, baseline_idx], axis=1, keepdims=True)

    # Interpolate onto common depth grid
    interp_kwargs = dict(axis=0, bounds_error=False, fill_value=np.nan)
    lfp_power = interp1d(unique_depths, lfp_power, **interp_kwargs)(DEPTH_GRID)
    lfp_mean = interp1d(unique_depths, lfp_mean, **interp_kwargs)(DEPTH_GRID)

    # Current source density: -conductivity * second spatial derivative of the potential
    # (gaussian_filter1d does not handle NaNs so only use the grid points with data)
    has_data = ~np.isnan(lfp_mean[:, 0])
    csd = np.full(lfp_mean.shape, np.nan)
    if np.sum(has_data) >= 3:
        lfp_smooth = gaussian_filter1d(lfp_mean[has_data], CSD_SMOOTH, axis=0)
        spacing = np.median(np.diff(DEPTH_GRID)) * 1e-6  # m
        d2 = np.full(lfp_smooth.shape, np.nan)
        d2[1:-1] = (lfp_smooth[:-2] - 2 * lfp_smooth[1:-1] + lfp_smooth[2:]) / spacing ** 2
        csd[has_data] = -CONDUCTIVITY * d2 * 1e-3  # A/m^3 -> uA/mm^3

    power_all.append(lfp_power)
    lfp_all.append(lfp_mean)
    csd_all.append(csd)

# Average over recordings
print(f'\nIncluded {len(lfp_all)} recordings')
power_avg = np.nanmean(np.stack(power_all), axis=0)
lfp_avg = np.nanmean(np.stack(lfp_all), axis=0) * 1e6  # V -> uV
csd_avg = np.nanmean(np.stack(csd_all), axis=0)
np.savez(join(save_path, 'LFP_depth_profile_MOs.npz'), time=time_ax, depth=DEPTH_GRID,
         power=np.stack(power_all), lfp=np.stack(lfp_all), csd=np.stack(csd_all))

# Smoothing over depth and time (NaN-aware: normalized gaussian convolution)
sigma_bins = [PLOT_SMOOTH / np.median(np.diff(DEPTH_GRID)),
              PLOT_SMOOTH_TIME / np.median(np.diff(time_ax))]
valid = ~np.isnan(lfp_avg)
lfp_smooth = gaussian_filter(np.where(valid, lfp_avg, 0), sigma_bins)
weights = gaussian_filter(valid.astype(float), sigma_bins)
lfp_smooth = np.where(weights > 0.5, lfp_smooth / weights, np.nan)

# %% Plot
extent = [time_ax[0], time_ax[-1], DEPTH_GRID[0], DEPTH_GRID[-1]]
f, ax = plt.subplots(figsize=(3, 2.2), dpi=dpi)
clim = np.nanpercentile(np.abs(lfp_smooth), 99)
im = ax.imshow(lfp_smooth, aspect='auto', origin='lower', extent=extent, cmap='RdBu_r',
               vmin=-clim, vmax=clim, interpolation='bilinear')
ax.axvline(0, color='k', lw=0.75, ls='--')
ax.set(xlabel='Time from 5-HT stimulation (s)',
       ylabel=f'Depth from L2/3 and L5 boundary ($\mu$m)',
       xticks=np.arange(-TIME_WIN[0], TIME_WIN[1] + 1),
       xlim=[-1, 3], ylim=YLIM)
cbar = f.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label('Evoked LFP ($\mu$V)', rotation=270, labelpad=10)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'LFP', f'LFP_depth_profile_{REGION[0]}.jpg'), dpi=600)
plt.savefig(join(fig_path, 'LFP', f'LFP_depth_profile_{REGION[0]}.pdf'))
plt.show()
