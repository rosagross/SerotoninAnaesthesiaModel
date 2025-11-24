# %%
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
S_baseline = 0
G = [2]
targets_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S}_G{G}.csv'))
targets_baseline_df = pd.read_csv(os.path.join(analysed_data_dir, f'serotonin_targets_state_analysis_S{S_baseline}_G{G}.csv'))

# %%

# compute delta downstate probability 
mean_baseline = targets_baseline_df.groupby(['region_name'], as_index=False).mean()['p_down']
targets_baseline_df['p_down_mean'] = targets_baseline_df.groupby(['region_name', 'G'])['p_down'].transform('mean')
targets_baseline_df['p_down_delta'] = targets_baseline_df['p_down'] - targets_baseline_df['p_down_mean']
targets_df['p_down_delta'] = (targets_df['p_down'].values - targets_baseline_df['p_down_mean'].values - targets_baseline_df['p_down_delta'].values)

# plot 
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
plt.savefig(os.path.join(figure_dir, f'trajectory_serotonin_targets_G{G}_S{S}.png'), bbox_inches="tight")
plt.show()
