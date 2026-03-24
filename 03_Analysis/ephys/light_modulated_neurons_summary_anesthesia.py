#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap
from updown_functions import paths, figure_style, load_subjects

# Settings
N_BINS = 30
MIN_NEURONS = 2
AP = [2, -1.5, -3.5]

# Get paths
fig_path, save_path = paths()

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype and subject number
subjects = load_subjects(only_expression=False)
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['expression'] == 1]
wt_neurons = all_neurons[all_neurons['expression'] == 0]

# Calculate percentage modulated neurons
all_mice = ((sert_neurons.groupby(['subject']).sum()['modulated']
             / sert_neurons.groupby(['subject']).size() * 100).to_frame().reset_index())
all_mice['expression'] = 1
wt_mice = ((wt_neurons.groupby(['subject']).sum()['modulated']
            / wt_neurons.groupby(['subject']).size() * 100).to_frame().reset_index())
wt_mice['expression'] = 0
all_mice = pd.concat((all_mice, wt_mice), ignore_index=True)
all_mice = all_mice.rename({0: 'perc_mod'}, axis=1)



# %% Plot percentage mod neurons
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)

f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
#sns.stripplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=3,
#              palette=[colors['sert'], colors['wt']], ax=ax1, jitter=0.2)
sns.swarmplot(x='expression', y='perc_mod', data=all_mice, order=[1, 0], size=3,
              color='k', legend=None, ax=ax1)
ax1.set(xticks=[0, 1], xticklabels=['Yes', 'No'], ylabel='Modulated neurons (%)', ylim=[-1, 100],
        xlabel='Expression in DRN', yticks=[0, 50, 100], xlim=[-0.3, 1.5])

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'no_expression', 'light_mod_summary.pdf'))

