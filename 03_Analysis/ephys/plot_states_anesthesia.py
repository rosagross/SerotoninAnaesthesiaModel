#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname, split
import pandas as pd
from matplotlib.patches import Rectangle
from updown_functions import figure_style, paths, load_subjects, add_significance

EXPRESSION = True

# Paths
fig_path, save_path = paths()

# Load in data
p_state_df = pd.read_csv(join(save_path, 'updown_states_anesthesia.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    p_state_df.loc[p_state_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
p_state_df = p_state_df[p_state_df['expression'] == int(EXPRESSION)]

# Generate p-values for each region and timepoint
p_values = dict()
for region in p_state_df['region'].unique():
    p_values[region] = []
    for time in p_state_df['time'].unique():
        _, p = ttest_1samp(p_state_df.loc[(p_state_df['region'] == region) & (p_state_df['time'] == time), 'p_down_bl'], 0)
        p_values[region].append(p)
    p_values[region] = np.array(p_values[region])


# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 8, figsize=(7, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
titles = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
for i, region in enumerate(regions):

    axs[i].add_patch(Rectangle((0, -0.25), 1, 0.75, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0, 0], lw=1, ls='--', color='grey')
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down_bl',
                 color='k', errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    add_significance(p_state_df['time'].unique(), p_values=p_values[region], ax=axs[i], y_pos=0.39)
    axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.15, 0.4],
               yticks=[-0.15, 0, 0.4], yticklabels=[-15, 0, 40])
    #n_sub = len(np.unique(p_state_df.loc[p_state_df['region'] == region, 'subject']))
    if i == 0:
        #axs[i].text(2, -0.22, f'n={n_sub} mice', ha='center', va='center')
        axs[i].set_ylabel('Down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.26, -0.26], color='k', lw=0.5)
        axs[i].text(1, -0.275, '2s', ha='center', va='top')
    elif i == len(regions)-1:
        #axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    else:
        #axs[i].text(3, -0.22, f'n={n_sub}', ha='center', va='center')
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
    
plt.subplots_adjust(left=0.08, bottom=0.1, right=1, top=0.8, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
if EXPRESSION:
    plt.savefig(join(fig_path, 'ephys_updown_states', 'p_down_state_anesthesia.pdf'))
else:
    plt.savefig(join(fig_path, 'no_expression', 'p_down_state_anesthesia_no_expression.pdf'))
plt.show()

