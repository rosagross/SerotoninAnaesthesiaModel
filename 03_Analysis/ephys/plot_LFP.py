
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (load_lfp, paths, remap, query_ephys_sessions, load_passive_opto_times,
                              combine_regions, get_neuron_qc, init_one, figure_style)
from iblatlas.atlas import AllenAtlas
colors, dpi = figure_style()
ba = AllenAtlas()
one = init_one()
_, save_path = paths()

# Settings
NEURON_QC = True
OPTO_PULSE_NR = 40
TIME_WIN = [5, 5]
PLOT_CORTEX = ['MOs', 'VIS', 'RSPd']

# Query sessions
rec = query_ephys_sessions(only_expression=False, one=one)


for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date}')

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

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Get cortex channels
    beryl_regions = remap(channels['acronym'], dest='Beryl')
    cortex_channels = np.where(np.isin(beryl_regions, PLOT_CORTEX))[0]
    if len(cortex_channels) == 0:
        print(f'No cortex channels found for {eid}')
        continue

    # Filter neurons that pass QC
    if NEURON_QC:
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba, force_rerun=False)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Load in LFP of cortex around a laser pulse
    lfp_arr, lfp_timestamps = load_lfp(
        eid, probe=probe, time_start=opto_times[OPTO_PULSE_NR] - TIME_WIN[0],
        time_end=opto_times[OPTO_PULSE_NR] + TIME_WIN[1], channels=cortex_channels)
    channel_depths = channels['z'][cortex_channels] * 1e6

    sampling_rate = 1 / np.median(np.diff(lfp_timestamps))
    sos = butter(4, 300, btype='highpass', fs=sampling_rate, output='sos')
    lfp_arr = sosfiltfilt(sos, lfp_arr, axis=-1)

    
    # Plot LFP
    f, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    for ch in range(0, lfp_arr.shape[0], 10):
        ax.plot(lfp_timestamps - opto_times[OPTO_PULSE_NR], lfp_arr[ch, :]*1e5 + channel_depths[ch],
                color='k', lw=0.2)
        ax.text(TIME_WIN[1] + 0.5, channel_depths[ch], f'{channels["acronym"][cortex_channels[ch]]}',
                fontsize=6, color='k', ha='left', va='center')
    ax.set(xlabel='Time from laser pulse (s)', ylabel='Depth ($\mu$m)', xlim=[-TIME_WIN[0], TIME_WIN[1]],
           xticks=[-TIME_WIN[0], 0, TIME_WIN[1]], xticklabels=[-TIME_WIN[0], 0, TIME_WIN[1]],
           title=f'{subject}, {date}, {probe}')
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(f'C:\\Users\\Guido1\\Figures\\SerotoninAnesthesia\\{subject}_{date}_{probe}.jpg',
                dpi=600)
    plt.close(f)

    
    

    
    


    
