import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'paper_panel'))

# load data
S = 40 
G = [2]
data_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S{S}_G[0, 1, 2, 3]_RateAdj1.csv'))
baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S0_G{G}_RateAdj1.csv'))

# summarize regions according to higher level regions
frontal = ['Medial prefrontal cortex', 'Orbitofrontal cortex', 'Secondary motor cortex'] # = 'Frontal'
sensory = ['Piriform', 'Visual cortex'] # = 'Sensory'
midbrain = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Superior colliculus'] # = 'Midbrain'
relevant_area_IDs = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
data_df['combined_region'] = data_df['region_name']
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(frontal))] = 'Frontal cortex' 
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(sensory))] = 'Sensory cortex' 
data_df['combined_region'].iloc[np.where(data_df['region_name'].isin(midbrain))] = 'Midbrain' 
data_df = data_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()
baseline_df['combined_region'] = baseline_df['region_name']
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(frontal))] = 'Frontal cortex' 
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(sensory))] = 'Sensory cortex' 
baseline_df['combined_region'].iloc[np.where(baseline_df['region_name'].isin(midbrain))] = 'Midbrain' 
baseline_df = baseline_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()
serotonin_df = data_df[data_df['G'] == 2]

# compute the delta downstate for all regions 
baseline_df['p_down_mean'] = baseline_df.groupby(['combined_region', 'G'])['p_down'].transform('mean')
baseline_df['p_down_delta'] = baseline_df['p_down'] - baseline_df['p_down_mean']
serotonin_df['p_down_delta'] = (serotonin_df['p_down'].values - baseline_df['p_down_mean'].values - baseline_df['p_down_delta'].values)

# %% Plot 

colors, dpi = figure_style()
f, axs = plt.subplots(1, 2, figsize=(2.5, 1.75), sharey=True, gridspec_kw={'width_ratios': [3, 2.5]})

region = 'Frontal cortex'
serotonin_window = [0.5,1]

# first axis: trajectory of one area
stimulation = serotonin_df[serotonin_df['combined_region'] == region]
no_stimulation = baseline_df[baseline_df['combined_region'] == region]

axs[0].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
axs[0].plot([-1, 3], [0, 0], ls='--', color='grey')
axs[0].plot([0, 2], [-0.27, -0.27], color='k', lw=0.5, clip_on=False)

sns.lineplot(data=stimulation, x='time', y='p_down_delta',
                color=colors['stim'], errorbar='se', err_kws={'lw': 0}, ax=axs[0], label='Stimulation')
sns.lineplot(data=no_stimulation, x='time', y='p_down_delta',
                color=colors['no-stim'], errorbar='se', ax=axs[0], label='No stimulation')
axs[0].set(xlabel='Time (s)', title=region, ylim=[-0.265, 0.5],
            yticks=[-0.25, 0, 0.5, 1], yticklabels=[-25, 0, 50, 100])
axs[0].set_title(region)
axs[0].legend(title=None, loc='upper right', bbox_to_anchor=(1.3, 1.05))    
axs[0].set_ylabel(u'Δ down state probability (%)', labelpad=0)
axs[0].get_xaxis().set_visible(False)
sns.despine(trim=True, bottom=True, ax=axs[0])
axs[0].text(1, -0.29, '2s', ha='center', va='top')

# second axis: bar plots of all other regions 
regions = ['Amygdala',  'Tail of the striatum', 'Sensory cortex', 
           'Hippocampus', 'Thalamus','Midbrain']
region_labels = ['Amyg',  'Str', 'Sens', 
           'Hipp', 'Thal','Mid']
stimulation_window_df = serotonin_df[serotonin_df['time'].between(serotonin_window[0],serotonin_window[1])]
stimulation_window_df = stimulation_window_df.groupby(['combined_region', 'session'], as_index=False).mean()

sns.barplot(data=stimulation_window_df, y='p_down_delta', x='combined_region',
            ax=axs[1], order=regions,
            color='grey', clip_on=False)

axs[1].set_xticklabels(region_labels, rotation=-45, ha='left')
axs[1].set_title('Stimulation')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].get_yaxis().set_visible(False)

sns.despine(trim=True, left=True, ax=axs[1])
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.tight_layout(h_pad=-10, w_pad=0)
plt.savefig(os.path.join(figure_dir, f'downstate_replication_S{S}_G{G}.pdf'), dpi=600)

