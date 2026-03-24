#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from updown_functions import paths, figure_style, high_level_regions
colors, dpi = figure_style()

# Get paths
fig_path, save_path = paths()

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['high_level_region'] = high_level_regions(all_neurons['region'], input_atlas='Beryl',
                                                      only_vis=True)
all_neurons = all_neurons[all_neurons['high_level_region'] != 'root']
grouped_neurons = all_neurons.groupby('high_level_region').size().reset_index()
grouped_neurons = grouped_neurons.rename(columns={0: 'n_neurons'})
grouped_neurons = grouped_neurons.sort_values(by='n_neurons', ascending=False)

f, ax = plt.subplots(figsize=(2, 1.75), dpi=dpi)
sns.barplot(grouped_neurons, x='n_neurons', y='high_level_region', color='grey')
ax.set(ylabel='', xlabel='# of recorded neurons', xticks=[0, 200, 400, 600])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'n_neurons.pdf'))
plt.show()