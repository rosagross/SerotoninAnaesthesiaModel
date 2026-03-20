# %%
import pandas as pd
import numpy as np
import os 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import group_regions, figure_style
colors, dpi = figure_style()

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'Figures', 'Results'))

# load data
S = 40 
G = [2]
data_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S{S}_G{G}_RateAdj1_UPDATE.csv'))
baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'state_analysis_bins_S0_G{G}_RateAdj1_UPDATE.csv'))

data_df = data_df[data_df['G'] == 2]
baseline_df = baseline_df[baseline_df['G'] == 2]

# summarize regions according to higher level regions
data_df = group_regions(data_df)
data_df = data_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean(numeric_only=True)
baseline_df = group_regions(baseline_df)
baseline_df = baseline_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean(numeric_only=True)
serotonin_df = data_df.copy()

# compute the delta downstate for all regions 
baseline_df['p_down_mean'] = baseline_df.groupby(['combined_region', 'G', 'session'])['p_down'].transform('mean')
baseline_df['p_down_delta'] = baseline_df['p_down'] - baseline_df['p_down_mean']
serotonin_df['p_down_delta'] = (serotonin_df['p_down'].values - baseline_df['p_down_mean'].values - baseline_df['p_down_delta'].values)

# plot in this order
regions = ['Frontal cortex', 'Striatum',  'Amygdala', 'Piriform cortex', 'Visual cortex',
           'Hippocampus', 'Thalamus', 'Midbrain']
titles = ['Frontal\ncortex', 'Striatum',  'Amygdala', 'Piriform\ncortex', 'Visual\ncortex',
          'Hippocampus', 'Thalamus', 'Midbrain']

# %% Plot: delta downstate probability 

f, axs = plt.subplots(1, 8, figsize=(5.25, 1.75), sharey=True, dpi=dpi)

for i, region in enumerate(regions):
    
    axs[i].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    
    stimulation = serotonin_df[serotonin_df['combined_region'] == region]
    sns.lineplot(data=stimulation, x='time', y='p_down_delta',
                 color='k', errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    axs[i].set(xlabel='Time (s)', title=titles[i], ylim=[-0.355, 0.5],
               yticks=[-0.35, 0, 0.5, 1], yticklabels=[-35, 0, 50, 100])
    
    if i == 0:
        axs[i].set_ylabel('Down state probability (%)', labelpad=0)
        axs[i].get_xaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=axs[i])
        axs[i].plot([0, 2], [-0.35, -0.35], color='k', lw=0.5)
        axs[i].text(1, -0.37, '2s', ha='center', va='top')
    else:
        axs[i].get_yaxis().set_visible(False)
        axs[i].axis('off')

plt.subplots_adjust(left=0.08, bottom=0.1, right=1, top=0.8, wspace=0, hspace=0.4)
#plt.tight_layout(w_pad=)
plt.savefig(os.path.join(figure_dir, f'p_delta_downstate_S{S}_G{G}.pdf'))
plt.show()

# %% bar plot & t-test
region_labels = ['Frontal', 'Amyg',  'Str', 'Sens', 'Hipp', 'Thal','Mid']
serotonin_window = [0.5,1]
window_df = serotonin_df[serotonin_df['time'].between(serotonin_window[0], serotonin_window[1])]

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi, sharey=True)

sns.barplot(data=window_df, y='p_down_delta', x='combined_region',
            ax=axs, order=regions, errorbar='se', 
            color='grey', clip_on=False)

axs.set_xticklabels(region_labels, rotation=-45, ha='left')
axs.set_title('DOWN state upon serotonin release')
axs.set_xlabel('')
axs.set_ylabel(u'Δ down state probability (%)')
axs.text(-0.25, 0.57, f'***', fontsize=6)
axs.text(0.75, 0.9, f'***', fontsize=6)
axs.text(1.75, 0.3, f'n.s.', fontsize=6)
axs.text(2.75, 0.7, f'***', fontsize=6)
axs.text(3.75, 0.1, f'n.s.', fontsize=6)
axs.text(4.75, 0.63, f'***', fontsize=6)
axs.text(5.75, 0.2, f'***', fontsize=6)

sns.despine(trim=True, ax=axs)
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.tight_layout(h_pad=-10, w_pad=0)
#plt.savefig(os.path.join(figure_dir, f'downstate_replicationS{S}_G{G}.pdf'), dpi=600)

# %% t test

for i, region in enumerate(regions):
    sample = window_df[window_df['combined_region']==region]['p_down_delta']
    t, p = stats.ttest_1samp(sample, 0, axis=0)
    print(region, '{:f}'.format(p))
    

# %% Plot transitions

colors, dpi = figure_style()
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)


for i, region in enumerate(regions):
    
    region_data = serotonin_df[serotonin_df['combined_region'] == region]
    region_data = region_data[region_data['time'] > -0.95]
    
    axs[i].axvspan(0, 1, alpha=0.1, color='royalblue')
    sns.lineplot(data=region_data, x='time', y='p_state_change_down',
                 color=colors['suppressed'], errorbar='se', ax=axs[i], label='to down')
    sns.lineplot(data=region_data, x='time', y='p_state_change_up',
                 color=colors['enhanced'], errorbar='se', ax=axs[i], label='to up')
    axs[i].set(ylabel='P(state change)', xlabel='Time (s)', title=region, xticks=[-1, 0, 1, 2, 3, 4])
    axs[i].set_title(region, fontsize=9)
    #handles, labels = axs[i].get_legend_handles_labels()
    #plt.legend(handles, labels,bbox_to_anchor=(-0, 0))
    axs[i].legend(loc='best')
    if i >0:
        axs[i].get_legend().remove()
    
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(os.path.join(figure_dir, f'downstate_transitions_S{S}_G{G}.pdf'), dpi=600)
plt.show()