#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from latenzy import latenzy
from brainbox.io.one import SpikeSortingLoader
from updown_functions import (paths, query_ephys_sessions,
                              get_neuron_qc, init_one, load_passive_opto_times)
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = init_one()

# Settings
USE_DUR = 1.2
OVERWRITE = True
NEURON_QC = True
_, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    latency_df = pd.DataFrame()
else:
    latency_df = pd.read_csv(join(save_path, 'neuron_type.csv'))
    rec = rec[~rec['pid'].isin(latency_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date}')

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

    # Get opto times
    opto_times, _ = load_passive_opto_times(eid, one=one)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    latency = np.full(clusters_pass.shape[0], np.nan)
    for n, neuron_id in enumerate(clusters_pass):
        try:
            latency[n], _ = latenzy(spikes.times[spikes.clusters == neuron_id],
                                    opto_times,
                                    USE_DUR,
                                    jitter_size=2,
                                    peak_alpha=0.15
                                    )
        except Exception as err:
            print(err)
            continue

    # Add to dataframe
    latency_df = pd.concat((latency_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
        'neuron_id': clusters_pass, 'latency': latency
        })))

    # Save output
    latency_df.to_csv(join(save_path, 'latency.csv'), index=False)
