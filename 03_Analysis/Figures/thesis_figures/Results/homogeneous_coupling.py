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
1) all 14 areas, homogeneous connectivity, stimulation vs no stimulation
2) all 14 areas, plot homogeneous & normal coupling together in one plot
3) plot the difference (in serotonin window) between both coupling types and sort
   it by incoming coupling stregth.
4) does coupling strength still have an important effect here? 
There is no selective projections anymore so now only the DRN projection should be relevant 
G = 0 vs G = 3
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure5'))
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'data', 'firing_rates'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))

# load data
homogen_C_allG_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[2, 3]_RateAdj1_C.csv'))
homogen_C_baseline_allG_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[2, 3]_RateAdj1_C.csv'))
heterogen_C_allG_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[0, 1, 2, 3]_RateAdj1.csv'))
heterogen_C_baseline_allG_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[0, 1, 2, 3]_RateAdj1.csv'))

homogen_C_baseline_allG_df['p_down_mean'] = homogen_C_baseline_allG_df.groupby(['region_name', 'G', 'session'])['p_down'].transform('mean')
homogen_C_baseline_allG_df['p_down_delta'] = homogen_C_baseline_allG_df['p_down'] - homogen_C_baseline_allG_df['p_down_mean']
homogen_C_allG_df['p_down_delta'] = (homogen_C_allG_df['p_down'].values - homogen_C_baseline_allG_df['p_down_mean'].values - homogen_C_baseline_allG_df['p_down_delta'].values)

heterogen_C_baseline_allG_df['p_down_mean'] = heterogen_C_baseline_allG_df.groupby(['region_name', 'G', 'session'])['p_down'].transform('mean')
heterogen_C_baseline_allG_df['p_down_delta'] = heterogen_C_baseline_allG_df['p_down'] - heterogen_C_baseline_allG_df['p_down_mean']
heterogen_C_allG_df['p_down_delta'] = (heterogen_C_allG_df['p_down'].values - heterogen_C_baseline_allG_df['p_down_mean'].values - heterogen_C_baseline_allG_df['p_down_delta'].values)

heterogen_C_df = heterogen_C_allG_df[heterogen_C_allG_df['G']==2]
heterogen_C_baseline_df = heterogen_C_baseline_allG_df[heterogen_C_baseline_allG_df['G']==2]
homogen_C_df = homogen_C_allG_df[homogen_C_allG_df['G']==2]
homogen_C_baseline_df = homogen_C_baseline_allG_df[homogen_C_baseline_allG_df['G']==2]

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)
all_acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'VIS']


# %% Plot 14 areas - with BASELINE

colors, dpi = figure_style()
fig, axs = plt.subplots(2, 7, sharex=True, sharey=False, figsize=(7,2))

for i, ax in enumerate(axs.flatten()):
    ax.axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    ax.plot([-1, 3], [0, 0], ls='--', color='grey')

    homogen_stim = homogen_C_df[homogen_C_df['region_name']==atlas[i][0]]
    homogen_baseline = homogen_C_baseline_df[homogen_C_baseline_df['region_name']==atlas[i][0]]

    sns.lineplot(data=homogen_stim, x='time', y='p_down_delta',
                    color=colors['stim'], errorbar='se', err_kws={'lw': 0}, ax=ax, label='Stimulation')
    sns.lineplot(data=homogen_baseline, x='time', y='p_down_delta', err_kws={'lw': 0},
                    color=colors['no-stim'], errorbar='se', ax=ax, label='No stim.')
    ax.set(ylim=[-0.40, 1], yticks=[-0.4, 0, 0.4, 0.8, 1], yticklabels=[-40, 0, 40, 80, 100], xlim=[-0.8, 4])
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


# %% Plot 14 areas - Heterogeneous vs Homogeneous connectivity

colors, dpi = figure_style()
fig, axs = plt.subplots(2, 7, sharex=True, sharey=False, figsize=(7,2))

for i, ax in enumerate(axs.flatten()):
    ax.axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    ax.plot([-1, 3], [0, 0], ls='--', color='grey')

    homogen_stim = homogen_C_df[homogen_C_df['region_name']==atlas[i][0]]
    heterogen_stim = heterogen_C_df[heterogen_C_df['region_name']==atlas[i][0]]

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
plt.savefig(os.path.join(figure_dir, f'heterogen_vs_homogen_connectivity.pdf'), bbox_inches="tight")

# %% Plot the difference between heterogeneous and homogeneous connectivity 

connect_matrix = pd.read_csv(os.path.join(analysed_data_dir, 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median_thresh-0.005.csv' ))
connect_matrix.drop('Unnamed: 0', inplace=True, axis=1)
connect_matrix = np.array(connect_matrix)
incoming_projections = np.mean(connect_matrix, axis=0)
all_regions = np.unique(homogen_C_df['region_name'])
serotonin_window = [0.5,1]

heterogen_window = heterogen_C_df[heterogen_C_df['time'].between(serotonin_window[0], serotonin_window[1])]
heterogen_grouped = heterogen_window.groupby(['region_name','session'], as_index=False).mean()
homogen_window = homogen_C_df[homogen_C_df['time'].between(serotonin_window[0], serotonin_window[1])]
homogen_grouped = homogen_window.groupby(['region_name','session'], as_index=False).mean()

diff_df = heterogen_grouped.copy()
cols = ['p_down', 'p_down_delta']
diff_df[cols] = heterogen_grouped[cols] - homogen_grouped[cols]

for i, region in enumerate(all_regions):

    # get the projection density
    diff_df.loc[diff_df['region_name'] == region, 'incoming_projections'] = incoming_projections[i]
    diff_df.loc[diff_df['region_name'] == region, 'abr_region'] = all_acronyms[i]


# %% Plot barplot sorted by incoming projections

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 1.75), dpi=dpi)

# exclude regions that do not show UP and DOWN states at all 
regions_UP = ['Hipp', 'MRN', 'PAG', 'SC']
regions_plot_df = diff_df[~diff_df['abr_region'].isin(regions_UP)]
plot_order = regions_plot_df.groupby(['abr_region'], as_index=False).mean().sort_values(by='incoming_projections', ascending=True).abr_region.values

sns.barplot(regions_plot_df, x='abr_region', y='p_down_delta', color=colors['grey'], errorbar='se', order=plot_order)
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='x', which='major', pad=0)
ax1.set_xlabel('')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-45)
sns.despine(trim=True, ax=ax1)
ax1.set_ylabel(u'heterogen. - homogen. \nΔ down state prob. (%)')
plt.savefig(os.path.join(figure_dir, f'heterogen_vs_homogen_barplot.pdf'), bbox_inches="tight")

# %% Plot uncoupled vs homogeneous coupling G=3

uncoupled_df = heterogen_C_allG_df[heterogen_C_allG_df['G']==0]
homogen_coupled_df = homogen_C_allG_df[homogen_C_allG_df['G']==3]

# add regions abbreviation and group in window
all_regions = np.unique(homogen_coupled_df['region_name'])
serotonin_window = [0.5,1]

uncoupled_window = uncoupled_df[uncoupled_df['time'].between(serotonin_window[0], serotonin_window[1])]
uncoupled_grouped = uncoupled_window.groupby(['region_name'], as_index=False).mean()
homogen_window = homogen_coupled_df[homogen_coupled_df['time'].between(serotonin_window[0], serotonin_window[1])]
homogen_grouped = homogen_window.groupby(['region_name'], as_index=False).mean()

uncoupled_coupled_diff_df = uncoupled_grouped.copy()
cols = ['p_down', 'p_down_delta']
uncoupled_coupled_diff_df[cols] = homogen_grouped[cols] - uncoupled_grouped[cols]

for i, region in enumerate(all_regions):

    # get the projection density
    uncoupled_coupled_diff_df.loc[uncoupled_coupled_diff_df['region_name'] == region, 'abr_region'] = all_acronyms[i]

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
(
     so.Plot(uncoupled_coupled_diff_df, x='drn_connect', y='p_down_delta')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)

uncoupled_coupled_diff_df['color'] = [colors[i] for i in uncoupled_coupled_diff_df['abr_region']]

for i in uncoupled_coupled_diff_df.index:
    so.Dot(pointsize=0)
    txt = ax1.text(uncoupled_coupled_diff_df.loc[i, 'drn_connect'], uncoupled_coupled_diff_df.loc[i, 'p_down_delta'], uncoupled_coupled_diff_df.loc[i, 'abr_region'],
             color=uncoupled_coupled_diff_df.loc[i, 'color'], fontsize=8.5, fontweight='bold')
    r, p = pearsonr(uncoupled_coupled_diff_df['drn_connect'], uncoupled_coupled_diff_df['p_down_delta'])
    ax1.text(0.09, 0.24, f'r = {r:.2f}', fontsize=6)
    ax1.text(0.1, 0.3, f'**', fontsize=6)

ax1.set_yticks([-0.2, 0, 0.2, 0.4])
ax1.set_yticklabels([-20, 0, 20, 40])
ax1.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
ax1.set_xlabel('Projection density (from DRN)')
ax1.set_ylabel(u' G0v3 Δ down state prob. (%)')
sns.despine(trim=True)
plt.savefig(os.path.join(figure_dir, f'homogenG2_vs_uncoupled.pdf'), bbox_inches="tight")
plt.show()
