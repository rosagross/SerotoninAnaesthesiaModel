# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from stim_functions import figure_style, group_regions
import seaborn.objects as so
from scipy.stats import pearsonr, f_oneway
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
colors, dpi = figure_style()

'''
1) direct effect
2) indrect effect visual cortex
3) indirect effect all areas
4) traces of selected areas
'''

# set directories
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'analysed_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '03_Analysis', 'Figures', 'Results'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, '02_Data', 'connectivity'))

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
serotonin_window = [0, 1]

# Define logistic function
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# %%
# summarize regions according to higher level regions
all_acronyms = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'VIS']
all_regions = np.unique(baseline_df['region_name'])
for i, region in enumerate(all_regions):

    # get the projection density
    baseline_df.loc[baseline_df['region_name'] == region, 'abr_region'] = all_acronyms[i]
    serotonin_df.loc[serotonin_df['region_name'] == region, 'abr_region'] = all_acronyms[i]

serotonin_df = group_regions(serotonin_df)
serotonin_df = serotonin_df.dropna()
baseline_df = group_regions(baseline_df)
baseline_df = baseline_df.dropna()


# %% Plot direct effect - grouped regions

# average delta p_down 
window_df = serotonin_df[serotonin_df['time'].between(serotonin_window[0],serotonin_window[1])] 
uncoupled_stimulation_df = window_df[(window_df['G']==0) & (window_df['S']==40)]
average_prob_df = uncoupled_stimulation_df.groupby(['combined_region', 'abr_region_combined', 'time'], as_index=False).mean(numeric_only=True)
average_prob_df = average_prob_df.groupby(['combined_region', 'abr_region_combined'], as_index=False).max(numeric_only=True)
average_prob_df['color'] = [colors[i] for i in average_prob_df['abr_region_combined']]

f, ax = plt.subplots(1, 1,figsize=(1.75, 1.75), dpi=dpi)
(
     so.Plot(average_prob_df, x='drn_connect', y='p_down_delta')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax)
     .plot()
)


for i in average_prob_df.index:
    ax.text(average_prob_df.loc[i, 'drn_connect'], average_prob_df.loc[i, 'p_down_delta'], average_prob_df.loc[i, 'abr_region_combined'],
             color=average_prob_df.loc[i, 'color'], fontsize=8.5, fontweight='bold')
    #ax.set(yticks=[0.25, 0.5, 0.75, 1], xticks=[-0.0, 0.1, 0.2])

r, p = pearsonr(average_prob_df['drn_connect'], average_prob_df['p_down_delta'])
print(p)
#ax.text(0.09, 0.36, f'r = {r:.2f}', fontsize=6)
ax.text(0.1, 0.68, f'**', fontsize=12, ha='center')

ax.set_title('Uncoupled network (G=0)')

plt.yticks([0, 0.45, 0.9])
ax.set_yticklabels([0, 45, 90])
plt.xticks([0, 0.1, 0.2])
ax.set_xticklabels([0, 0.1, 0.2])
plt.xlabel('Projection density (from DRN)')
plt.ylabel('Down state probability (%)')
sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'combined_regions_drnConnect_vs_pDown_G0_S40.pdf'), bbox_inches="tight")
plt.show()

# %% Plot indirect effect - grouped regions

# average delta p_down
window_df = serotonin_df[serotonin_df['time'].between(serotonin_window[0], serotonin_window[1])]
uncoupled_stimulation_df = window_df[(window_df['G'] == 3) & (window_df['S'] == 40)]
average_prob_df = uncoupled_stimulation_df.groupby(['combined_region', 'abr_region_combined', 'time'], as_index=False).mean(numeric_only=True)
average_prob_df = average_prob_df.groupby(['combined_region', 'abr_region_combined'], as_index=False).max(numeric_only=True)
average_prob_df['color'] = [colors[i] for i in average_prob_df['abr_region_combined']]

f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
    so.Plot(average_prob_df, x='drn_connect', y='p_down_delta')
    .add(so.Dot(pointsize=0))
    .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
    .on(ax)
    .plot()
)

for i in average_prob_df.index:
    ax.text(average_prob_df.loc[i, 'drn_connect'], average_prob_df.loc[i, 'p_down_delta'],
            average_prob_df.loc[i, 'abr_region_combined'],
            color=average_prob_df.loc[i, 'color'], fontsize=8.5, fontweight='bold')
# ax.set(yticks=[0.25, 0.5, 0.75, 1], xticks=[-0.0, 0.1, 0.2])
r, p = pearsonr(average_prob_df['drn_connect'], average_prob_df['p_down_delta'])
print(p)
#ax.text(0.09, 0.36, f'r = {r:.2f}', fontsize=6)
ax.text(0.1, 0.72, f'*', fontsize=12, ha='center')

plt.yticks([0, 0.45, 0.9])
ax.set_yticklabels([0, 45, 90])
plt.xticks([0, 0.1, 0.2])
ax.set_xticklabels([0, 0.1, 0.2])
ax.set_title('Connected network (G=3)')
plt.xlabel('Projection density (from DRN)')
plt.ylabel('Down state probability (%)')
sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'combined_regions_drnConnect_vs_pDown_G3_S40.pdf'), bbox_inches="tight")
plt.show()


# %% Plot indirect effect

# choose which area to plot 
plot_area = 'Frontal cortex'
region_df = serotonin_df[(serotonin_df['combined_region'] == plot_area) & (serotonin_df['S'] == 40)]

# define window of serotonin effect 
region_window_df = region_df[region_df['time'].between(serotonin_window[0], serotonin_window[1])]
mean_time_df = region_window_df.groupby(['G', 'time'], as_index=False).mean(numeric_only=True)
plot_df = mean_time_df.groupby(['G'], as_index=False).max(numeric_only=True)

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)
ax1.plot(plot_df['G'], plot_df['p_down_delta'], marker='o', zorder=1, color='k', markersize=4)
ax1.set_xlabel('Coupling strength (G)')
ax1.set_ylabel('Down state probability (%)')
ax1.set_title(plot_area)
ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.set_yticklabels([0, 20, 40, 60])
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels([0, 1, 2, 3])

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(os.path.join(figure_dir, f'indirect_effect_{plot_area.replace(" ", "_")}_S40.pdf'), bbox_inches="tight")
plt.show()


# %% Plot delta

plot_df = pd.DataFrame()
for region in serotonin_df['combined_region'].unique():
    region_df = serotonin_df[(serotonin_df['combined_region'] == region) & (serotonin_df['S'] == 40)]
    region_window_df = region_df[region_df['time'].between(serotonin_window[0], serotonin_window[1])]
    mean_time_df = region_window_df.groupby(['G', 'time'], as_index=False).mean(numeric_only=True)
    max_time_df = mean_time_df.groupby(['G'], as_index=False).max(numeric_only=True)
    delta = np.abs(max_time_df.loc[max_time_df['G'] == 3, 'p_down_delta'].values
                   - max_time_df.loc[max_time_df['G'] == 0, 'p_down_delta'].values)
    plot_df = pd.concat((plot_df, pd.DataFrame(index=[plot_df.shape[0]], data={
        'region_name':region, 'delta': delta})))
plot_df = plot_df.sort_values(by='delta', ascending=False)

f, ax1 = plt.subplots(figsize=(2, 1.75), dpi=dpi)
sns.barplot(data=plot_df, x='delta', y='region_name')
ax1.set(ylabel='', xlabel=u'Δ down state', xticks=[0, 0.2, 0.4], xticklabels=[0, 0.2, 0.4])

sns.despine(trim=True)
plt.tight_layout()
plt.show()


# %%
# Compute received projection density

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
                                                     'combined_region':region_window_df['combined_region'].values[0], 'abr_region_combined':region_window_df['abr_region_combined'].values[0],
                                                     'g_diff' : g_diff, 'targeted':incoming_projections[i]}, index=[i])))

drn_order = corr_df.sort_values(by='drn_connect', ascending=False).abr_region.values


# %%
# combined regions
# Plot region coloured by received projection density
corr_df_grouped = corr_df.groupby(['combined_region', 'abr_region_combined'], as_index=False).mean(numeric_only=True)
corr_df_grouped['color'] = [colors[i] for i in corr_df_grouped['abr_region_combined']]
# integrate info about coupling (projections targeting the area)
corr_df_grouped.loc[corr_df_grouped['targeted']<0.05, 'target_grouped'] = '0.0-0.05'
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.05, 0.1), 'target_grouped'] = '0.05-0.1'
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.1, 0.15), 'target_grouped'] = '0.1-0.15'
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.15, 0.2), 'target_grouped'] = '0.15-0.2'

colors_connect = sns.color_palette('Dark2')[0:4]
corr_df_grouped.loc[corr_df_grouped['targeted']<0.05, 'target_color'] = rgb2hex(colors_connect[0])
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.05, 0.1), 'target_color'] = rgb2hex(colors_connect[1])
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.1, 0.15), 'target_color'] = rgb2hex(colors_connect[2])
corr_df_grouped.loc[corr_df_grouped['targeted'].between(0.15, 0.2), 'target_color'] = rgb2hex(colors_connect[3])



colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
(
     so.Plot(corr_df_grouped, x='drn_connect', y='g_diff')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)

for i in corr_df_grouped.index:
    so.Dot(pointsize=0)
    txt = ax1.text(corr_df_grouped.loc[i, 'drn_connect'], corr_df_grouped.loc[i, 'g_diff'], corr_df_grouped.loc[i, 'abr_region_combined'],
             color=corr_df_grouped.loc[i, 'target_color'], fontsize=8.5, fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
    r, p = pearsonr(corr_df_grouped['drn_connect'], corr_df_grouped['g_diff'])
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
ax1.set_ylabel(u'Coupling G 0 versus 3')
sns.despine(trim=True)
lgd = ax1.legend(title='Received projection density', handles=handles, labels=labels,
                bbox_to_anchor=(0.3, 1, 1., .102),labelspacing=0.2)
plt.savefig(os.path.join(figure_dir, f'combined_regions_coupling_effect_S40.pdf'), bbox_inches="tight")
plt.show()

# %% 
# Specific regions
# Plot region coloured by received projection density

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
ax1.set_ylabel(u'Coupling G 0 versus 3')
sns.despine(trim=True)
lgd = ax1.legend(title='Received projection density', handles=handles, labels=labels,
                bbox_to_anchor=(0.3, 1, 1., .102),labelspacing=0.2)
plt.savefig(os.path.join(figure_dir, f'coupling_effect_S40.pdf'), bbox_inches="tight")
plt.show()

# %% Plot traces

colors, dpi = figure_style()
f, axs = plt.subplots(1, 4, figsize=(5, 2.0), dpi=dpi, sharey=True)

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
plt.show()
