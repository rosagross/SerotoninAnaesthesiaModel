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
serotonin_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S40_G[0, 1, 2, 3]_RateAdj1.csv'))
baseline_df = pd.read_csv(os.path.join(analysed_data_dir, 'state_analysis_bins_S0_G[0, 1, 2, 3]_RateAdj1.csv'))

# compute the delta downstate probability
baseline_df['p_down_mean'] = baseline_df.groupby(['region_name', 'G'])['p_down'].transform('mean')
baseline_df['p_down_delta'] = baseline_df['p_down'] - baseline_df['p_down_mean']
serotonin_df['p_down_delta'] = (serotonin_df['p_down'].values - baseline_df['p_down_mean'].values - baseline_df['p_down_delta'].values)

# choose which area to plot 
plot_area = 'Visual cortex'
region_df = serotonin_df[serotonin_df['region_name'] == plot_area]

# define window of serotonin effect 
serotonin_window = [0.5,1]
region_window_df = region_df[region_df['time'].between(serotonin_window[0],serotonin_window[1])] 

# %% Plot 

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
plt.savefig(os.path.join(figure_dir, f'indirect_effect_{plot_area.replace(" ", "_")}_S40.png'), bbox_inches="tight")
plt.show()

