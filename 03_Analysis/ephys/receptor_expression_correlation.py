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
from os.path import join, realpath, dirname, split
from scipy.stats import pearsonr
from updown_functions import paths, figure_style, load_subjects, high_level_regions, remap
colors, dpi = figure_style()

MIN_NEURONS = 5

# Paths
fig_path, data_path = paths()

# Load in expression
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
#expr_df['region'] = high_level_regions(remap(expr_df['acronym']))
expr_df['region'] = remap(expr_df['acronym'])

# Get expression value per region per receptor
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# Load in neural data
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
#ephys_data['region'] = high_level_regions(ephys_data['region'])
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ephys_data.loc[ephys_data['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]

# Calculate summary per neuron
ephys_summary = ephys_data[(ephys_data['expression'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index']].groupby('region').mean()['mod_index'].to_frame()
ephys_summary['n_neurons'] = ephys_data[(ephys_data['expression'] == 1) & (ephys_data['modulated'] == 1)].groupby(
    'region').size()
ephys_summary = ephys_summary.reset_index()
ephys_summary['abs_mod_index'] = np.abs(ephys_summary['mod_index'])

# Drop regions with too few neurons
ephys_summary = ephys_summary[ephys_summary['n_neurons'] >= MIN_NEURONS]

# %% receptors

receptor = '5-HT1f'
plot_data = pd.merge(ephys_summary,
                     expression_mean[expression_mean['receptor'] == receptor],
                     on=['region'])

fig, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

r, p = pearsonr(plot_data['mod_index'], plot_data['expression_energy'])
print(f'{receptor}; p = {np.round(p, 2)}')
ax.scatter(plot_data['mod_index'], plot_data['expression_energy'], color='k', s=10, zorder=1)
if p < 0.05:
    slope, intercept = np.polyfit(plot_data['mod_index'], plot_data['expression_energy'], 1)
    x_fit = np.linspace(plot_data['mod_index'].min(), plot_data['mod_index'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='k', label='_nolegend_', zorder=0)
    if p < 0.001:
        ax.text(-0.25, 9.5, '***', fontsize=12, ha='center', va='center')
    elif p < 0.01:
        ax.text(-0.25, 9.5, '**', fontsize=12, ha='center', va='center')
    else:
        ax.text(-0.25, 9.5, '*', fontsize=12, ha='center', va='center')
ax.set(xlabel='Modulation index', ylabel=f'{receptor} expression',
       xlim=[-0.5, 0], xticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0], xticklabels=[-0.5, -0.4, -0.3, -0.2, -0.1, 0],
       yticks=[0, 2, 4, 6, 8, 10])

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, f'{receptor}_receptor.pdf'))
plt.show()
