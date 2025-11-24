# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style
import seaborn.objects as so
from scipy.stats import pearsonr, f_oneway
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches

'''
1) direct effect
2) indrect effect visual cortex
3) indirect effect all areas
4) traces of selected areas
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure3'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir))

# load data
serotonin_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[0, 1, 2, 3]_RateAdj1.csv'))
baseline_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[0, 1, 2, 3]_RateAdj1.csv'))
connect_matrix = pd.read_csv(os.path.join(atlas_dir, 'MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median_thresh-0.005.csv' ))
connect_matrix.drop('Unnamed: 0', inplace=True, axis=1)
connect_matrix = np.array(connect_matrix)

# compute the delta downstate probability
baseline_df['p_down_mean'] = baseline_df.groupby(['region_name', 'G', 'session'])['p_down'].transform('mean')
baseline_df['p_down_delta'] = baseline_df['p_down'] - baseline_df['p_down_mean']
serotonin_df['p_down_delta'] = (serotonin_df['p_down'].values - baseline_df['p_down_mean'].values - baseline_df['p_down_delta'].values)

# define window of serotonin effect 
serotonin_window = [0.5,1]

# %% Plot direct effect

colors, dpi = figure_style()

# average delta p_down 
acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'VIS']
window_df = serotonin_df[serotonin_df['time'].between(serotonin_window[0],serotonin_window[1])] 
uncoupled_stimulation_df = window_df[(window_df['G']==0) & (window_df['S']==40)]
average_prob_df = uncoupled_stimulation_df.groupby(['region_name'], as_index=False).mean()
average_prob_df['abr_region'] = acronyms
average_prob_df['color'] = [colors[i] for i in average_prob_df['abr_region']]

f, ax = plt.subplots(1, 1,figsize=(2.5, 1.75), dpi=dpi)
(
     so.Plot(average_prob_df, x='drn_connect', y='p_down_delta')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax)
     .plot()
)


for i in average_prob_df.index:
    ax.text(average_prob_df.loc[i, 'drn_connect'], average_prob_df.loc[i, 'p_down_delta'], average_prob_df.loc[i, 'abr_region'],
             color=average_prob_df.loc[i, 'color'], fontsize=8.5, fontweight='bold')
    #ax.set(yticks=[0.25, 0.5, 0.75, 1], xticks=[-0.0, 0.1, 0.2])
    r, p = pearsonr(average_prob_df['drn_connect'], average_prob_df['p_down_delta'])
    ax.text(0.09, 0.36, f'r = {r:.2f}', fontsize=6)
    ax.text(0.1, 0.41, f'***', fontsize=6)
    

plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels([0, 20, 40, 60, 80])
plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
plt.xlabel('Projection density (from DRN)')
plt.ylabel(u'Δ down state probability (%)')
sns.despine(offset=2, trim=True)
plt.savefig(os.path.join(figure_dir, 'drnConnect_vs_pDown_G0_S40.pdf'), bbox_inches="tight")


# %% Plot indirect effect

# choose which area to plot 
plot_area = 'Visual cortex'
region_df = serotonin_df[serotonin_df['region_name'] == plot_area]

# define window of serotonin effect 
region_window_df = region_df[region_df['time'].between(serotonin_window[0],serotonin_window[1])] 

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.1, 1.75), dpi=dpi)

sns.despine(trim=False)
sns.barplot(region_window_df, x='G', y='p_down_delta', color=colors['grey'], errorbar='se')
ax1.tick_params(axis='x', which='major', pad=0)
ax1.set_xlabel('Coupling strength (G)')
ax1.set_ylabel(u'Δ down state probability (%)')
ax1.set_title(plot_area)
ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.set_yticklabels([0, 20, 40, 60])
ax1.set_xticklabels(ax1.get_xticklabels())
ax1.tick_params(axis='x', which='major', pad=1.75)
plt.savefig(os.path.join(figure_dir, f'indirect_effect_{plot_area.replace(" ", "_")}_S40.pdf'), bbox_inches="tight")
plt.show()

values = region_window_df.groupby(['G','session'],as_index=False).mean()
f_statistic, p_value = f_oneway(values[values['G']==0]['p_down_delta'], values[values['G']==1]['p_down_delta'], 
                                values[values['G']==2]['p_down_delta'], values[values['G']==3]['p_down_delta'])

# Print the results
print("F statistic:", f_statistic)
print("p-value:", p_value)

# %% Plot 

window_df['drn_connect'] = np.round(window_df['drn_connect'], 4)
all_regions = np.unique(window_df['region_name'])
incoming_projections = np.mean(connect_matrix, axis=0)

corr_df = pd.DataFrame()

for i, region in enumerate(all_regions):
    region_window_df = window_df[window_df['region_name']==region]

    # calculate the difference between G0 and G3
    g0 = region_window_df[region_window_df['G']==0]['p_down_delta'].mean()
    g3 = region_window_df[region_window_df['G']==3]['p_down_delta'].mean()
    g_diff = g3 - g0
    r, p = pearsonr(region_window_df['G'], region_window_df['p_down_delta'])

    # collect the r and p values
    drn_value = np.unique(region_window_df['drn_connect'])
    corr_df = pd.concat((corr_df, pd.DataFrame(data={'region':region, 'r':r, 'p':p, 'drn_connect':drn_value[0], 'abr_region':acronyms[i],
                                                     'g_diff' : g_diff, 'targeted':incoming_projections[i]}, index=[i])))

drn_order = corr_df.sort_values(by='drn_connect', ascending=False).abr_region.values

# integrate info about coupling (projections targeting the area)
corr_df.loc[corr_df['targeted']<0.05, 'target_grouped'] = '0.0-0.05'
corr_df.loc[corr_df['targeted'].between(0.05, 0.1), 'target_grouped'] = '0.05-0.1'
corr_df.loc[corr_df['targeted'].between(0.1, 0.15), 'target_grouped'] = '0.1-0.15'
corr_df.loc[corr_df['targeted'].between(0.15, 0.2), 'target_grouped'] = '0.15-0.2'

colors_connect = sns.color_palette('Dark2')[0:4]
corr_df.loc[corr_df['targeted']<0.05, 'target_color'] = rgb2hex(colors_connect[0])
corr_df.loc[corr_df['targeted'].between(0.05, 0.1), 'target_color'] = rgb2hex(colors_connect[1])
corr_df.loc[corr_df['targeted'].between(0.1, 0.15), 'target_color'] = rgb2hex(colors_connect[2])
corr_df.loc[corr_df['targeted'].between(0.15, 0.2), 'target_color'] = rgb2hex(colors_connect[3])

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
(
     so.Plot(corr_df, x='drn_connect', y='g_diff')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)

for i in corr_df.index:
    so.Dot(pointsize=0)
    txt = ax1.text(corr_df.loc[i, 'drn_connect'], corr_df.loc[i, 'g_diff'], corr_df.loc[i, 'abr_region'],
             color=corr_df.loc[i, 'target_color'], fontsize=8.5, fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
    r, p = pearsonr(corr_df['drn_connect'], corr_df['g_diff'])
    ax1.text(0.09, 0.3, f'r = {r:.2f}', fontsize=6)
    ax1.text(0.09, 0.24, f'p = {p:.2f}', fontsize=6)
    
labels = ['0.0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2']
handles = []
for color in colors_connect:
    dot = Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6)
    handles.append(dot)

patches = [mpatches.Patch(color=rgb2hex(color), label=label) for color, label in zip(colors_connect, labels)]
ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.set_yticklabels([0, 20, 40, 60])
ax1.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
ax1.set_xlabel('Projection density (from DRN)')
ax1.set_ylabel(u' G0v3 Δ down state prob. (%)')
sns.despine(trim=True)
lgd = ax1.legend(title='Received projection density', handles=handles, labels=labels,
                bbox_to_anchor=(0.3, 1, 1., .102),labelspacing=0.2)
plt.savefig(os.path.join(figure_dir, f'coupling_effect_S40.pdf'), bbox_inches="tight")
plt.show()

# %% Plot traces

colors, dpi = figure_style()
f, axs = plt.subplots(1, 4, figsize=(6, 1.75), dpi=dpi, sharey=True)

regions_traces = ['Visual cortex', 'Medial prefrontal cortex', 'Thalamus', 'Amygdala']
regions_abbrev = ['VIS', 'mPFC', 'Thal', 'Amyg']
for i, region in enumerate(regions_traces):
   

    data_df = serotonin_df[(serotonin_df['region_name'] == region)]
    
    sns.lineplot(data=data_df, x='time', y='p_down_delta', hue='G',
                 color=colors['suppressed'], errorbar='se', 
                 err_kws={'lw': 0}, ax=axs[i], palette='Dark2')
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].axvspan(0, 1, alpha=0.25, color='royalblue', lw=0)
    if i == 0:
        axs[i].set(ylabel=u'Δ down state probability (%)')
        axs[i].plot([0, 2], [-0.49, -0.49], color='k', lw=0.5)
        axs[i].text(1, -0.55, '2s', ha='center', va='top')
        plt.ylim([-0.5, 0.9])
        plt.yticks([-0.5, 0, 0.5, 0.9])
        axs[i].set_yticklabels([-50, 0, 50, 90])
        axs[i].get_xaxis().set_visible(False)
        axs[i].legend(title='G', bbox_to_anchor=(0., 0.6, 1., .102), labelspacing=0.1)
    else:
        axs[i].legend('')
        axs[i].axis('off')
        axs[i].get_yaxis().set_visible(False)
        axs[i].get_xaxis().set_visible(False)

    sns.despine(trim=True, bottom=True, ax=axs[i])
    axs[i].set_title(regions_abbrev[i])
    plt.tight_layout(w_pad=0)
        
plt.savefig(os.path.join(figure_dir, f'couplingstrength_trajectories_S40.pdf'), bbox_inches="tight")

