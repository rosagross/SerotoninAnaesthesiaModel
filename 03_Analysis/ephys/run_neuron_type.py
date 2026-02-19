#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (paths, remap, query_ephys_sessions, high_level_regions,
                              get_neuron_qc, init_one)
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()

# Settings
OVERWRITE = False
NEURON_QC = True
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    neuron_type = pd.DataFrame()
else:
    neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
    rec = rec[~rec['pid'].isin(neuron_type['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date}')

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.peakToTrough'])
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Filter neurons that pass QC and exclude artifact neurons
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    # Get data collection
    collections = one.list_collections(eid)
    if f'alf/{probe}/pykilosort' in collections:
        collection = f'alf/{probe}/pykilosort'
    else:
        collection = f'alf/{probe}'

    # Load in waveforms
    data = one.load_datasets(eid, datasets=['_phy_spikes_subset.waveforms', '_phy_spikes_subset.spikes',
                                            '_phy_spikes_subset.channels'],
                             collections=[collection]*3)[0]
    waveforms, wf_spikes, wf_channels = data[0], data[1], data[2]

    # Convert to uV
    waveforms = waveforms * 1000

    # Loop over clusters
    spike_amp, pt_ratio = np.full(clusters_pass.shape[0], np.nan), np.full(clusters_pass.shape[0], np.nan)
    pt_subtract, peak_to_trough = np.full(clusters_pass.shape[0], np.nan), np.full(clusters_pass.shape[0], np.nan)
    spike_width, rp_slope = np.full(clusters_pass.shape[0], np.nan), np.full(clusters_pass.shape[0], np.nan)
    rc_slope, firing_rate = np.full(clusters_pass.shape[0], np.nan), np.full(clusters_pass.shape[0], np.nan)
    for n, neuron_id in enumerate(clusters_pass):

        # Get mean waveform of channel with max amplitude
        n_waveforms = waveforms[spikes.clusters[wf_spikes] == neuron_id].shape[0]
        if n_waveforms == 0:
            continue

        mean_wf_ch = np.mean(waveforms[spikes.clusters[wf_spikes] == neuron_id], axis=0)
        mean_wf_ch = (mean_wf_ch
                      - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
        mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))]
        wf_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        spike_amp[n] = np.abs(np.min(mean_wf) - np.max(mean_wf))

        # Get peak-to-trough ratio
        pt_ratio[n] = np.max(mean_wf) / np.abs(np.min(mean_wf))

        # Get peak minus through
        pt_subtract[n] = np.max(mean_wf) - np.abs(np.min(mean_wf))

        # Get part of spike from trough to first peak after the trough
        peak_after_trough = np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)
        repolarization = mean_wf[np.argmin(mean_wf):np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)]

        # Get spike width in ms
        x_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        peak_to_trough[n] = ((np.argmax(mean_wf) - np.argmin(mean_wf)) / 30000) * 1000
        spike_width[n] = ((peak_after_trough - np.argmin(mean_wf)) / 30000) * 1000

        # Get repolarization slope
        if spike_width[n] <= 0.08:
            continue
        else:
            rp_slope[n], _, = np.polyfit(x_time[np.argmin(mean_wf):peak_after_trough],
                                         mean_wf[np.argmin(mean_wf):peak_after_trough], 1)

        # Get recovery slope
        rc_slope[n], _ = np.polyfit(x_time[peak_after_trough:], mean_wf[peak_after_trough:], 1)

        # Get firing rate
        firing_rate[n] = (np.sum(spikes['clusters'] == neuron_id)
                          / np.max(spikes['times']))

    cluster_regions = remap(clusters.acronym[clusters_pass])
    cluster_high_level = high_level_regions(cluster_regions)
    neuron_type = pd.concat((neuron_type, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
        'allen_acronym': clusters.acronym[clusters_pass], 'beryl_region': cluster_regions,
        'high_level_region': cluster_high_level, 'neuron_id': clusters_pass,
        'spike_width': spike_width, 'spike_amp': spike_amp, 'peak_to_trough': peak_to_trough,
        'rc_slope': rc_slope, 'rp_slope': rp_slope, 'pt_subtract': pt_subtract,
        'pt_ratio': pt_ratio
        })))

    # Determine neuron type by simple threshold
    neuron_type.loc[neuron_type['spike_width'] >= 0.4, 'neuron_type'] = 'PYR'
    neuron_type.loc[neuron_type['spike_width'] < 0.4, 'neuron_type'] = 'INT'

    # Save output
    neuron_type.to_csv(join(save_path, 'neuron_type.csv'), index=False)
