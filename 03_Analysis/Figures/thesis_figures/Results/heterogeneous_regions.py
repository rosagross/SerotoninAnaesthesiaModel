# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from scipy.stats import pearsonr
from stim_functions import figure_style

'''
Plot homogeneous regions vs. not homogeneous regions
1) traces of all areas
2) barplot showing the grouped regions, hue= heterogen vs. homogen
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure6'))
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))

# load firing rate  data 
file_dir = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_B_sessions"
file_name = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_B_sessions_0"
start = 0*1000
stop = 8*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]


# load state data
homogen_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[2]_RateAdj1_B.csv'))
homogen_baseline_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[2]_RateAdj1_B.csv'))
heterogen_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[2]_RateAdj1_UPDATE.csv'))
heterogen_baseline_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[2]_RateAdj1_UPDATE.csv'))

# compute delta down state probability
homogen_baseline_df['p_down_mean'] = homogen_baseline_df.groupby(['region_name', 'G', 'session'])['p_down'].transform('mean')
homogen_baseline_df['p_down_delta'] = homogen_baseline_df['p_down'] - homogen_baseline_df['p_down_mean']
homogen_df['p_down_delta'] = (homogen_df['p_down'].values - homogen_baseline_df['p_down_mean'].values - homogen_baseline_df['p_down_delta'].values)

heterogen_baseline_df['p_down_mean'] = heterogen_baseline_df.groupby(['region_name', 'G', 'session'])['p_down'].transform('mean')
heterogen_baseline_df['p_down_delta'] = heterogen_baseline_df['p_down'] - heterogen_baseline_df['p_down_mean']
heterogen_df['p_down_delta'] = (heterogen_df['p_down'].values - heterogen_baseline_df['p_down_mean'].values - heterogen_baseline_df['p_down_delta'].values)

# join dataframes
homogen_df['pattern'] = 'homogen'
heterogen_df['pattern'] = 'heterogen'
data_df = pd.concat((homogen_df, heterogen_df))

# group into lager regions
frontal = ['Medial prefrontal cortex', 'Orbitofrontal cortex', 'Secondary motor cortex'] # = 'Frontal'
sensory = ['Piriform', 'Visual cortex'] # = 'Sensory'
midbrain = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Superior colliculus'] # = 'Midbrain'

grouped_df = data_df
grouped_df['combined_region'] = data_df['region_name']
grouped_df['combined_region'].iloc[np.where(data_df['region_name'].isin(frontal))] = 'Frontal cortex' 
grouped_df['combined_region'].iloc[np.where(data_df['region_name'].isin(sensory))] = 'Sensory cortex' 
grouped_df['combined_region'].iloc[np.where(data_df['region_name'].isin(midbrain))] = 'Midbrain' 
grouped_df = grouped_df.groupby(['combined_region', 'session', 'time', 'pattern'], as_index=False).mean()
regions_plot = ['Frontal cortex', 'Amygdala', 'Tail of the striatum', 'Sensory cortex', 
           'Hippocampus', 'Thalamus','Midbrain']

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)
all_acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'VIS']
all_regions = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'Vis']

# %% Plot the firing rates of all areas to show homogeneity

colors, dpi = figure_style()
fig, axes = plt.subplots(2, 7, sharex=True, sharey=False, figsize=(7,2))

for i, ax in enumerate(axes.flatten()):
    
    ax.plot(example_rates[1,start:stop,i], label='I', color='#ff7f0e')
    ax.plot(example_rates[0,start:stop,i], label='E', color='#1f77b4')
    ax.set_title(all_regions[i], pad=0.5)
    ax.get_xaxis().set_visible(False)
    #ax.legend()
    ax.tick_params(axis='y', which='major', pad=0)

axes[1][0].plot([0*1000, 5*1000], [-3, -3], color='k', lw=0.8, clip_on=False)
axes[1][0].text(2.5*1000, -5, '5s', ha='center', va='top')
sns.despine(trim=True, bottom=True)
print(start, stop)
fig.text(0.08, 0.5, "firing rate (spike/s)", va='center', rotation='vertical')
plt.subplots_adjust(wspace=0.33, hspace=0.2)
plt.savefig(os.path.join(figure_dir, f'homogeneous_UP-DOWN_states_S0_G2.pdf'), dpi=600, bbox_inches="tight")

# %% Plot barplot 

# load in window 
serotonin_window = [0.5,1]
window_df = grouped_df[grouped_df['time'].between(serotonin_window[0], serotonin_window[1])]
regions_plot_df = window_df[window_df['combined_region'].isin(regions_plot)]

f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
sns.barplot(x='p_down_delta', y='combined_region', data=regions_plot_df, hue='pattern',
            ax=ax1, errorbar='se', order=regions_plot, palette="Dark2")

ax1.set(xlabel=u'Δ down state probability (%)', ylabel='', xlim=[0, 0.8])#, xticks=np.arange(0, 101, 20))
ax1.legend(frameon=False, bbox_to_anchor=(0.95, 0.45), prop={'size': 5}, title='',
           handletextpad=0.1)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(os.path.join(figure_dir, 'homogeneous_regions.pdf'))

# %% Plt traces of main regions

colors, dpi = figure_style()
fig, axs = plt.subplots(2, 7, sharex=True, sharey=False, figsize=(7,2))

for i, ax in enumerate(axs.flatten()):
    ax.axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    ax.plot([-1, 3], [0, 0], ls='--', color='grey')

    homogen_stim = homogen_df[homogen_df['region_name']==atlas[i][0]]
    heterogen_stim = heterogen_df[heterogen_df['region_name']==atlas[i][0]]

    sns.lineplot(data=homogen_stim, x='time', y='p_down_delta',
                    color=sns.color_palette('Dark2')[1], errorbar='se', err_kws={'lw': 0}, ax=ax, label='Homogeneous')
    sns.lineplot(data=heterogen_stim, x='time', y='p_down_delta', err_kws={'lw': 0},
                    color=sns.color_palette('Dark2')[2], errorbar='se', ax=ax, label='Heterogeneous')
    ax.set(ylim=[-0.50, 1], yticks=[-0.5, 0, 0.5, 1], yticklabels=[-50, 0, 50, 100], xlim=[-0.8, 4])
    ax.set_title(all_acronyms[i])
    ax.set_ylabel('', labelpad=0)
    ax.get_xaxis().set_visible(False)
    sns.despine(trim=True, bottom=True, ax=ax)
    
    if not i == 6:
        ax.legend('')
    else:
        ax.legend(title=None, loc='upper right', bbox_to_anchor=(1., 1.07))


    if not ((i==0) or (i==7)):
        ax.get_yaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=ax, left=True)

axs[1][0].plot([0, 2], [-0.5, -0.5], color='k', lw=0.8, clip_on=False)
axs[1][0].text(1, -0.55, '2s', ha='center', va='top')
fig.text(0.08, 0.5, u'Δ down state probability (%)', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0.4, wspace=0.)
plt.savefig(os.path.join(figure_dir, f'heterogen_vs_homogen_regions.pdf'), bbox_inches="tight")

