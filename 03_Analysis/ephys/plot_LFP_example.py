
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (load_lfp, paths, remap, load_passive_opto_times,
                              get_neuron_qc, init_one, figure_style)
from iblatlas.atlas import AllenAtlas
colors, dpi = figure_style()
ba = AllenAtlas()
one = init_one()
_, save_path = paths()

# Recording

SUBJECT = 'ZFM-05181'
DATE = '2023-01-10'
PROBE = 'probe00'
PID = '542aa93b-4876-439c-93f7-dd77263e55e8'
EID = '9ee8b642-9d7f-483f-93b4-5b2ed62c7653'

"""
SUBJECT = 'ZFM-04820'
DATE = '2022-09-13'
PROBE = 'probe00'
PID = 'cf08dda8-478f-4292-a06f-4c4dae9f8755'
EID = '5531e71f-8ab9-4c4e-8d5b-d92da838ee16'

SUBJECT = 'ZFM-05492'
DATE = '2023-02-07'
PROBE = 'probe00'
PID = '5e5d8b5c-0df0-446e-956f-c834769b2ab4'
EID = 'a59df7cb-bf0e-4160-993c-21de8aea6126'
"""
# Settings
REGION = 'MOs'
NEURON_QC = True
OPTO_PULSE_NR = 16
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

"""
# Filter LFP
sampling_rate = 1 / np.median(np.diff(lfp_timestamps))
sos = butter(4, 300, btype='highpass', fs=sampling_rate, output='sos')
lfp_arr = sosfiltfilt(sos, lfp_arr, axis=-1)
"""

# %% Plot
f, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)

for ch in range(0, lfp_arr.shape[0], 10):
    ax.plot(lfp_timestamps - opto_times[OPTO_PULSE_NR], lfp_arr[ch, :]*1e5 + channel_depths[ch],
            color='k', lw=0.1)
    ax.text(-TIME_WIN[0] - 0.5, channel_depths[ch], f'{channels["acronym"][cortex_channels[ch]]}',
            fontsize=6, color='k', ha='right', va='center')

for ch, channel in enumerate(cortex_channels):
    channel_clusters = np.where(clusters['channels'] == channel)[0]
    channel_spikes = spikes.times[np.isin(spikes.clusters, channel_clusters)]
    spike_times = channel_spikes - opto_times[OPTO_PULSE_NR]
    ax.vlines(spike_times, channel_depths[ch] - 10, channel_depths[ch] + 10,
              color='red', lw=0.5)

ax.axvspan(0, 1, color=colors['stim'], lw=0, alpha=0.5)

ax.set(xlabel='Time from 5-HT stimulation (s)', xlim=[-TIME_WIN[0], TIME_WIN[1]],
       xticks=np.arange(-TIME_WIN[0], TIME_WIN[1] + 1, 2), yticks=[])
plt.tight_layout()
sns.despine(trim=True, left=True)


    
    

    
    


    
