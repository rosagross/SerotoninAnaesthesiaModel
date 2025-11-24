# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

'''
Plot:
1) I with + current, show traces of all regions
2) difference trajectories for grouped regions
3) plot with all options averaged over all regions 
4) paradoxical effect: example region targeted I with - current (inhibiting I population leads to UP state)
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure4'))
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))

# load firing rate 
file_dir = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_J_sessions"
file_name = "14areas_G2.0_S40.0_thetaE-1_beta6RateAdj1_J_sessions_0"
start = 2*1000
stop = 8*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]
stim_times = pd.read_csv(os.path.join(frate_dir, file_dir, file_name + '_stimulation_times.csv')).to_numpy() - start

# load state analysis data 
S = 40
S_baseline = 0
G = [2]
targets_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S}_G{G}.csv'))
targets_baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S_baseline}_G{G}.csv'))

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)

# compute delta downstate probability per region
targets_baseline_df['p_down_mean'] = targets_baseline_df.groupby(['region_name', 'G', 'session', 'target_name'])['p_down'].transform('mean')
targets_baseline_df['p_down_delta'] = targets_baseline_df['p_down'] - targets_baseline_df['p_down_mean']
targets_df['p_down_delta'] = (targets_df['p_down'].values - targets_baseline_df['p_down_mean'].values - targets_baseline_df['p_down_delta'].values)

all_acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'Vis']

I_df = targets_df[targets_df['target_name']=='I+']
baseline_I_df = targets_baseline_df[targets_baseline_df['target_name']=='I+']
E_df = targets_df[targets_df['target_name']=='E-']

# %% Plot 1) traces of all regions when I targeted with + current 

colors, dpi = figure_style()
fig, axs = plt.subplots(2, 7, sharex=True, sharey=False, figsize=(7,2))

for i, ax in enumerate(axs.flatten()):
    ax.axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    ax.plot([-1, 3], [0, 0], ls='--', color='grey')

    stimulation = I_df[I_df['region_name']==atlas[i][0]]
    no_stimulation = baseline_I_df[baseline_I_df['region_name']==atlas[i][0]]

    sns.lineplot(data=stimulation, x='time', y='p_down_delta',
                    color=colors['stim'], errorbar='se', err_kws={'lw': 0}, ax=ax, label='Stimulation')
    sns.lineplot(data=no_stimulation, x='time', y='p_down_delta', err_kws={'lw': 0},
                    color=colors['no-stim'], errorbar='se', ax=ax, label='No stim.')
    ax.set(ylim=[-0.265, 0.3], yticks=[-0.25, 0, 0.3], yticklabels=[-25, 0, 30], xlim=[-0.8, 4])
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

axs[1][0].plot([0, 2], [-0.26, -0.26], color='k', lw=0.8, clip_on=False)
axs[1][0].text(1, -0.28, '2s', ha='center', va='top')
fig.text(0.08, 0.5, u'Δ down state probability (%)', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0.4, wspace=0.)
plt.savefig(os.path.join(figure_dir, f'I+_trajectories_G{G}_S{S}.pdf'), bbox_inches="tight")

# %% group regions

frontal = ['Medial prefrontal cortex', 'Orbitofrontal cortex', 'Secondary motor cortex'] # = 'Frontal'
sensory = ['Piriform', 'Visual cortex'] # = 'Sensory'
midbrain = ['Periaqueductal gray', 'Midbrain reticular nucleus', 'Superior colliculus'] # = 'Midbrain'
I_df['combined_region'] = I_df['region_name']
I_df['combined_region'].iloc[np.where(I_df['region_name'].isin(frontal))] = 'Frontal cortex' 
I_df['combined_region'].iloc[np.where(I_df['region_name'].isin(sensory))] = 'Sensory cortex' 
I_df['combined_region'].iloc[np.where(I_df['region_name'].isin(midbrain))] = 'Midbrain' 
I_df = I_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()
E_df['combined_region'] = E_df['region_name']
E_df['combined_region'].iloc[np.where(E_df['region_name'].isin(frontal))] = 'Frontal cortex' 
E_df['combined_region'].iloc[np.where(E_df['region_name'].isin(sensory))] = 'Sensory cortex' 
E_df['combined_region'].iloc[np.where(E_df['region_name'].isin(midbrain))] = 'Midbrain' 
E_df = E_df.groupby(['combined_region', 'S', 'session', 'G', 'time'], as_index=False).mean()

regions_plot = ['Frontal cortex', 'Amygdala', 'Tail of the striatum', 'Sensory cortex', 
           'Hippocampus', 'Thalamus','Midbrain']

# %% Plot 2) difference between E- and I+ in example regions

colors, dpi = figure_style()
fig, axs = plt.subplots(1, len(regions_plot), sharex=True, sharey=False, figsize=(8,1.5))

for i, ax in enumerate(axs):
    ax.axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    ax.plot([-1, 3], [0, 0], ls='--', color='grey')

    stimulation_I = I_df[I_df['combined_region']==regions_plot[i]][['p_down_delta', 'time']].reset_index()
    stimulation_E = E_df[E_df['combined_region']==regions_plot[i]][['p_down_delta', 'time']].reset_index()
    stimulation_diff = pd.DataFrame()
    stimulation_diff['p_down_delta'] = stimulation_E['p_down_delta'] - stimulation_I['p_down_delta']
    stimulation_diff['time'] = stimulation_E['time']

    sns.lineplot(data=stimulation_diff, x='time', y='p_down_delta',
                    color=colors['enhanced'], errorbar='se', err_kws={'lw': 0}, ax=ax, label='Stimulation')
    ax.set(ylim=[-0.50, 0.3], yticks=[-0.5, 0, 0.5, 1], yticklabels=[-50, 0, 50, 100])
    ax.set_title(regions_plot[i])
    ax.set_ylabel('', labelpad=0)
    ax.get_xaxis().set_visible(False)
    sns.despine(trim=True, bottom=True, ax=ax)
    ax.legend('')

    if not ((i==0) or (i==7)):
        ax.get_yaxis().set_visible(False)
        sns.despine(trim=True, bottom=True, ax=ax, left=True)

axs[0].plot([0, 2], [-0.5, -0.5], color='k', lw=0.8, clip_on=False)
axs[0].text(1, -0.55, '2s', ha='center', va='top')
fig.text(0.075, 0.5, u'E- vs. I+ Δ down state prob. (%)', va='center', rotation='vertical')
plt.savefig(os.path.join(figure_dir, f'difference_E-vsI+_G{G}_S{S}.pdf'), bbox_inches="tight")
plt.subplots_adjust(hspace=0.4)


# %% Plot 3)

colors, dpi = figure_style()
fig, ax = plt.subplots(1, 1, figsize=(1.7, 1.75), dpi=dpi)

plt.axvspan(0, 1, alpha=0.25, ymin=-0.25, color='royalblue', lw=0)
ax.plot([-1, 3], [0, 0], ls='--', color='grey')
ax.plot([0, 2], [-0.25, -0.25], color='k', lw=0.5, clip_on=False)
ax.text(1, -0.27, '2s', ha='center', va='top')
sns.lineplot(targets_df, x='time', y='p_down_delta', hue='target_name', 
             err_kws={'lw': 0}, palette=sns.color_palette('Dark2'), ax=ax)

ax.get_xaxis().set_visible(False)
ax.set_ylim([-0.25, 0.50])
ax.set_yticks([-0.25, 0, 0.25, 0.5])
ax.set_yticklabels([-25, 0, 25, 50])
ax.set_ylabel(u'Δ down state probability (%)')
ax.legend(title=None, loc='upper right', bbox_to_anchor=(0.95, 1.05))
sns.despine(trim=True, bottom=True)
plt.savefig(os.path.join(figure_dir, f'trajectory_serotonin_targets_G{G}_S{S}.pdf'), bbox_inches="tight")
plt.show()

# %% Plot 4)

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi, sharey=True)

region = 2

# plot rates
axs.axvspan(stim_times[0][0], stim_times[0][0]+1000, alpha=0.25, ymin=-0.25, color='royalblue', lw=0)
axs.plot(example_rates[1,:,region], label='I', color='#ff7f0e')
axs.plot(example_rates[0,:,region], label='E', color='#1f77b4')

# plot states
axs.set_xticks(ticks=[0, 2000, 4000, 6000])
axs.set_xticklabels(labels=np.array([0, 2, 4, 6]))
axs.set_yticks(ticks=[0, 25, 50])
axs.set_xlabel('time (s)')
axs.set_ylabel('firing rate (Hz)')
sns.despine(trim=True)
axs.set_title(atlas[region][0])

# save
plt.savefig(os.path.join(figure_dir, f'example_I-target_G{G}_S{S}.pdf'), bbox_inches="tight")
plt.show()

# %% Plot 5) barplot of other regions showing I-
targets_df_grouped = targets_df
targets_df_grouped['combined_region'] = targets_df['region_name']
targets_df_grouped['combined_region'].iloc[np.where(targets_df['region_name'].isin(frontal))] = 'Frontal cortex' 
targets_df_grouped['combined_region'].iloc[np.where(targets_df['region_name'].isin(sensory))] = 'Sensory cortex' 
targets_df_grouped['combined_region'].iloc[np.where(targets_df['region_name'].isin(midbrain))] = 'Midbrain' 
targets_df_grouped = targets_df_grouped.groupby(['combined_region', 'S', 'session', 'G', 'time', 'target_name'], as_index=False).mean()

serotonin_window = [0.5,1]
region_labels = ['Frontal', 'Amyg',  'Str', 'Sens', 'Hipp', 'Thal','Mid']
I_inhib_df = targets_df_grouped[targets_df_grouped['target_name']=='I-']
I_inhib_df = I_inhib_df[I_inhib_df['time'].between(serotonin_window[0],serotonin_window[1])]
I_inhib_df = I_inhib_df[I_inhib_df['combined_region'].isin(regions_plot)]

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi, sharey=True)

sns.barplot(data=I_inhib_df, y='p_down_delta', x='combined_region',
            ax=axs, order=regions_plot, errorbar='se', 
            color='grey', clip_on=False)

axs.set_xticklabels(region_labels, rotation=-45, ha='left')
axs.set_title('Excitatory stimulation of I')
axs.set_xlabel('')
axs.set_ylabel(u'Δ down state probability (%)')

sns.despine(trim=True, ax=axs)
plt.subplots_adjust(left=0.08, bottom=0.15, right=1, top=0.85, wspace=0, hspace=0.4)
plt.tight_layout(h_pad=-10, w_pad=0)
plt.savefig(os.path.join(figure_dir, f'paradoxical_effect_S{S}_G{G}.pdf'), dpi=600)




# %%
