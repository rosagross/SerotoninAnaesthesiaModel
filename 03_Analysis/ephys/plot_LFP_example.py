
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import seaborn as sns
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (load_lfp, paths, remap, load_passive_opto_times,
                              get_neuron_qc, init_one, figure_style)
from iblatlas.atlas import AllenAtlas
colors, dpi = figure_style()
ba = AllenAtlas()
one = init_one()
fig_path, save_path = paths()

# Recording
"""
REGION = 'MOs'
OPTO_PULSE_NR = 16
PLOT_EVERY_CH = 12
SUBJECT = 'ZFM-05181'
DATE = '2023-01-10'
PROBE = 'probe00'
PID = '542aa93b-4876-439c-93f7-dd77263e55e8'
EID = '9ee8b642-9d7f-483f-93b4-5b2ed62c7653'

REGION = 'VISpm'
OPTO_PULSE_NR = 40
PLOT_EVERY_CH = 8
SUBJECT = 'ZFM-05169'
DATE = '2022-11-16'
PROBE = 'probe00'
PID = '9c5e319e-09e8-4770-a806-46958bb464f1'
EID = '4bf62d04-e5a6-46db-add6-35e55db2de71'
"""
REGION = 'VISa'
OPTO_PULSE_NR = 14
SUBJECT = 'ZFM-04820'
DATE = '2022-09-15'
PROBE = 'probe00'
PID = '366c7c11-5509-443f-b07b-dabc9dcfd53e'
EID = '73fa004c-eac3-4b34-997a-9b4268990474'

# Settings
PLOT_N_CHANNELS = 10
LOW_PASS = True
LOW_PASS_FREQ = 20
NEURON_QC = True
TIME_WIN = [6, 6]

# Load in laser pulse times
opto_times, _ = load_passive_opto_times(EID, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Get cortex channels
beryl_regions = remap(channels['acronym'], dest='Beryl')
cortex_channels = np.where(beryl_regions == REGION)[0]

# Filter neurons that pass QC
if NEURON_QC:
    qc_metrics = get_neuron_qc(PID, one=one, ba=ba, force_rerun=False)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
else:
    clusters_pass = np.unique(spikes.clusters)
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Remove spikes in layer 1
layer1 = np.where(clusters.acronym == 'MOs1')[0]
spikes.times = spikes.times[~np.isin(spikes.clusters, layer1)]
spikes.clusters = spikes.clusters[~np.isin(spikes.clusters, layer1)]

# Load in LFP of cortex around a laser pulse
lfp_arr, lfp_timestamps = load_lfp(
    EID, probe=PROBE, time_start=opto_times[OPTO_PULSE_NR] - TIME_WIN[0],
    time_end=opto_times[OPTO_PULSE_NR] + TIME_WIN[1], channels=cortex_channels)
channel_depths = channels['z'][cortex_channels] * 1e6

if LOW_PASS:
    # Filter LFP
    sampling_rate = 1 / np.median(np.diff(lfp_timestamps))
    sos = butter(4, LOW_PASS_FREQ, btype='lowpass',
                 fs=sampling_rate, output='sos')
    lfp_arr = sosfiltfilt(sos, lfp_arr, axis=-1)

# %% Plot
f, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)

unique_depths, unique_channel_indices = np.unique(channel_depths, return_index=True)
target_depths = np.linspace(unique_depths[0], unique_depths[-1], PLOT_N_CHANNELS)
selected_channels = unique_channel_indices[
    np.abs(unique_depths[:, None] - target_depths).argmin(axis=0)
]


for ch in selected_channels:
    ax.plot(lfp_timestamps - opto_times[OPTO_PULSE_NR], lfp_arr[ch, :]*1e5 + channel_depths[ch],
            color='k', lw=0.3)
    ax.text(-TIME_WIN[0] - 0.5, channel_depths[ch], f'{channels["acronym"][cortex_channels[ch]]}',
            fontsize=6, color='k', ha='right', va='center')

for ch, channel in enumerate(cortex_channels):
    channel_clusters = np.where(clusters['channels'] == channel)[0]
    channel_spikes = spikes.times[np.isin(spikes.clusters, channel_clusters)]
    spike_times = channel_spikes - opto_times[OPTO_PULSE_NR]
    ax.vlines(spike_times, channel_depths[ch] - 10, channel_depths[ch] + 10,
              color='red', lw=0.5)

ax.axvspan(0, 1, color='royalblue', lw=0, alpha=0.5)

ax.set(xlabel='Time from 5-HT stimulation (s)', xlim=[-TIME_WIN[0], TIME_WIN[1]],
       xticks=np.arange(-TIME_WIN[0], TIME_WIN[1] + 1, 2), yticks=[])
plt.tight_layout()
sns.despine(trim=True, left=True)
plt.savefig(join(fig_path, 'LFP', f'LFP_example_{REGION}.pdf'))


    
    

    
    


    
