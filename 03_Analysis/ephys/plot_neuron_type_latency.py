# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:47:48 2026

@author: Guido
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from updown_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
fig_path, data_path = paths()
type_df = pd.read_csv(join(data_path, 'neuron_type.csv'))
latency_df = pd.read_csv(join(data_path, 'latency.csv'))
mod_df = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
merged_df = pd.merge(type_df, latency_df, on=['subject', 'date', 'eid', 'probe', 'pid', 'neuron_id'])
neurons_df = pd.merge(merged_df, mod_df, on=['subject', 'date', 'eid', 'probe', 'pid', 'neuron_id'])
neurons_df = neurons_df[~np.isin(neurons_df['high_level_region'], ['root', 'Midbrain', 'Amygdala', 'Thalamus'])]
neurons_df.loc[neurons_df['high_level_region'] == 'Frontal cortex', 'high_level_region'] = 'Cortex'
neurons_df.loc[neurons_df['high_level_region'] == 'Sensory cortex', 'high_level_region'] = 'Cortex'
neurons_df = neurons_df[neurons_df['modulated']]
regions = neurons_df['high_level_region'].unique()

# Do statistics
_, p = stats.ttest_ind(neurons_df.loc[(neurons_df['neuron_type'] == 'PYR') & ~np.isnan(neurons_df['latency']), 'latency'],
                       neurons_df.loc[(neurons_df['neuron_type'] == 'INT') & ~np.isnan(neurons_df['latency']), 'latency'])
print(p)

# %% Plot
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.histplot(data=neurons_df, x='spike_width', hue='neuron_type',
             hue_order=['INT', 'PYR'], palette=[colors['INT'], colors['PYR']], ax=ax, bins=36)
leg = ax.get_legend()
leg.set_title('')
ax.set(xlabel='Spike width (ms)', ylabel='Neuron count', xticks=[0, 0.5, 1], xticklabels=[0, 0.5, 1],
       xlim=[0, 1.02], yticks=[0, 30])

plt.tight_layout()
sns.despine(trim=True)

# %%
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.violinplot(data=neurons_df, x='neuron_type', y='latency', hue='neuron_type', order=['INT', 'PYR'],
               hue_order=['INT', 'PYR'], palette=[colors['INT'], colors['PYR']], ax=ax,
               linewidth=0.75, legend=None)
ax.text(0.5, 1, '*', fontsize=12, ha='center', va='center')
ax.set(ylabel='5-HT modulation latency (s)', xlabel='', yticks=[0, 0.5, 1], yticklabels=[0, 0.5, 1])

plt.tight_layout()
sns.despine(trim=True)

