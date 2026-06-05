#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from matplotlib.patches import Rectangle
from updown_functions import figure_style, paths, load_subjects, add_significance

EXPRESSION = True

# Paths
fig_path, save_path = paths()

# Load in data
p_state_df = pd.read_csv(join(save_path, 'updown_states_anesthesia_control.csv'))

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    p_state_df.loc[p_state_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
p_state_df = p_state_df[p_state_df['expression'] == int(EXPRESSION)]

# Average over mice first
p_state_df = p_state_df.groupby(['pid', 'time', 'region', 'jittered']).mean(numeric_only=True).reset_index()

# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 8, figsize=(7, 1.75), dpi=dpi, sharey=True)
regions = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
titles = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']

for i, region in enumerate(regions):

    # Do statistics
    p_values = np.zeros(p_state_df['time'].unique().shape[0])
    for k, time in enumerate(p_state_df['time'].unique()):
        stim = p_state_df.loc[(p_state_df['region'] == region) & (p_state_df['time'] == time)& (p_state_df['jittered'] == 0), 'p_down_up_bl'].values
        no_stim = p_state_df.loc[(p_state_df['region'] == region) & (p_state_df['time'] == time) & (p_state_df['jittered'] == 1), 'p_down_up_bl'].values
        _, p_values[k] = stats.ttest_rel(stim, no_stim)

    # Plot
    axs[i].add_patch(Rectangle((0, -0.4), 1, 1.5, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0, 0], lw=1, ls='--', color='grey')
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down_up_bl',
                 hue='jittered', hue_order=[0, 1], palette=[colors['stim'], colors['no-stim']],
                 errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    add_significance(p_state_df['time'].unique(), p_values=p_values, ax=axs[i], y_pos=0.48)
    axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.4, 0.5],
               yticks=[-0.4, 0, 0.5], yticklabels=[-40, 0, 50])
    #n_sub = len(np.unique(p_state_df.loc[p_state_df['region'] == region, 'subject']))
    if i == 0:
        axs[i].set_ylabel('Down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.41, -0.41], color='k', lw=0.5, clip_on=False)
        axs[i].text(1, -0.43, '2s', ha='center', va='top')
        axs[i].get_legend().remove()
    elif i == len(regions)-1:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        handles, previous_labels = axs[i].get_legend_handles_labels()
        axs[i].legend(title='', handles=handles, labels=['5-HT stimulation', 'No stimulation'])
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().remove()

plt.suptitle('Stimulation during UP state')
plt.subplots_adjust(left=0.08, bottom=0.1, right=1, top=0.75, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'ephys_updown_states', 'p_down_state_during_up.pdf'))
plt.show()


# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 8, figsize=(7, 1.75), dpi=dpi, sharey=True)
#axs = np.concatenate(axs)
regions = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
titles = ['Frontal cortex', 'Striatum', 'Amygdala', 'Piriform cortex', 'Visual cortex', 'Hippocampus',
           'Thalamus', 'Midbrain']
for i, region in enumerate(regions):

    # Do statistics
    p_values = np.zeros(p_state_df['time'].unique().shape[0])
    for k, time in enumerate(p_state_df['time'].unique()):
        stim = p_state_df.loc[(p_state_df['region'] == region) & (p_state_df['time'] == time)& (p_state_df['jittered'] == 0), 'p_down_down_bl'].values
        no_stim = p_state_df.loc[(p_state_df['region'] == region) & (p_state_df['time'] == time) & (p_state_df['jittered'] == 1), 'p_down_down_bl'].values
        _, p_values[k] = stats.ttest_ind(stim, no_stim)

    # Plot
    axs[i].add_patch(Rectangle((0, -0.25), 1, 0.75, color='royalblue', alpha=0.25, lw=0))
    axs[i].plot([-1, 4], [0, 0], lw=1, ls='--', color='grey')
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='p_down_down_bl',
                 hue='jittered', hue_order=[0, 1], palette=[colors['stim'], colors['no-stim']],
                 errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    add_significance(p_state_df['time'].unique(), p_values=p_values, ax=axs[i], y_pos=0.34)
    axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.15, 0.35],
               yticks=[-0.15, 0, 0.35], yticklabels=[-15, 0, 35])
    if i == 0:
        axs[i].set_ylabel('Down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.26, -0.26], color='k', lw=0.5)
        axs[i].text(1, -0.275, '2s', ha='center', va='top')
        axs[i].get_legend().remove()
    elif i == len(regions)-1:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        handles, previous_labels = axs[i].get_legend_handles_labels()
        axs[i].legend(title='', handles=handles, labels=['5-HT stimulation', 'No stimulation'])
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')
        axs[i].get_legend().remove()

plt.suptitle('Stimulation during DOWN state')
plt.subplots_adjust(left=0.08, bottom=0.1, right=1, top=0.75, wspace=0, hspace=0.4)
#plt.tight_layout(h_pad=-10, w_pad=1.08)
plt.savefig(join(fig_path, 'ephys_updown_states', 'p_down_state_during_down.pdf'))
plt.show()