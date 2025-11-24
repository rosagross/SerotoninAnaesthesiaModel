# %%

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import ssm 
from ssm.plots import gradient_cmap
from stim_functions import figure_style

"""
Replication of UP and DOWN state dynamic 
# three plots for one joint figure:
1) example UP/DOWN state of one area in G = 2
2) different UP/ DOWN state frequencies
3) synchrony 
"""

# set directories
frate_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'data', 'firing_rates'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'Writing', 'Figures', 'Results', 'Figure1'))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))
analysed_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir, os.pardir, 'ModelData', 'analysed_data'))

color_names = [
    "windows blue",
    "faded green"
    ]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

# load data 
file_dir = "14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions"
file_name = "14areas_G2.0_S0.0_thetaE-1_beta6RateAdj1_sessions_0"
start = 0*1000
stop = 8*1000
frate_E_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateE_{file_name}.csv'))
frate_I_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateI_{file_name}.csv'))
frate_A_sync = pd.read_csv(os.path.join(frate_dir, file_dir, f'frateA_{file_name}.csv')) 
example_rates = np.array((frate_E_sync, frate_I_sync, frate_A_sync))
example_rates = example_rates[:,start:stop,:]

# load state data
states_data = pd.read_csv(os.path.join(frate_dir, file_dir, file_name+'_states.csv'), index_col=0)

# choose area to plot 
region = 12
region_UP = 3

# load atlas with releveant regions for plotting
atlas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
atlas.drop(['Unnamed: 0'], inplace=True, axis=1)
atlas = np.array(atlas)
updown_regions = ['Amyg', 'mPFC', 'OLF', 'OFC', 'Pir', 'RSC', 'M2', 'Str', 'Thal', 'Vis']
up_regions = ['Hipp', 'MRN', 'PAG', 'SC']
all_regions = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSP', 'M2', 'SC', 'Str', 'Thal', 'Vis']

# %% plot one area UP-DOWN & one UP area

colors, dpi = figure_style()
f, axs = plt.subplots(1, 2, figsize=(3, 1.75), dpi=dpi, sharey=True)

# plot rates
axs[0].plot(example_rates[1,:,region], label='I', color='#ff7f0e')
axs[0].plot(example_rates[0,:,region], label='E', color='#1f77b4')

# plot states
data = states_data[states_data['region']==region].reset_index()
data = data.iloc[start:stop]
data['time'] = data['index']/1000
lim = 1.05 * abs(example_rates[1,:,region]).max()

axs[0].imshow(np.expand_dims(data['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[0].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[0].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[0].set_yticks(ticks=[0, 20, 40, 60])
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('firing rate (Hz)')
sns.despine(trim=True)
axs[0].set_title(atlas[region][0])


# plot rates UP state region
axs[1].plot(example_rates[1,:,region_UP], label='I', color='#ff7f0e')
axs[1].plot(example_rates[0,:,region_UP], label='E', color='#1f77b4')

data_UP = states_data[states_data['region']==region_UP].reset_index()
data_UP = data_UP.iloc[start:stop]
data_UP['time'] = data_UP['index']/1000

axs[1].imshow(np.expand_dims(data_UP['state'], axis=0),
           aspect="auto",
           cmap=cmap,
           alpha=0.2,
           # vmin=0,
           # vmax=len(colors)-1,
           extent=(0, len(example_rates[1,:,region]), -2, 60))

axs[1].set_xticks(ticks=[0, 2000, 4000, 6000, 8000, 10000])
axs[1].set_xticklabels(labels=[0, 2, 4, 6, 8, 10])
axs[1].set_yticks(ticks=[0, 20, 40, 60])
axs[1].set_xlabel('time (s)')
sns.despine(trim=True)
axs[1].set_title(atlas[region_UP][0])

plt.savefig(os.path.join(figure_dir, f'UPDOWN_{atlas[region][0]}-{atlas[region_UP][0]}S0_G2.pdf'), dpi=600, bbox_inches="tight")


# %% Load heterogeneity data

state_data = pd.read_csv(os.path.join(analysed_data_dir, 'totalduration_state_analysis_S0_G2RateAdj1.csv'))
up_areas = ['Midbrain reticular nucleus', 'Hippocampus', 'Superior colliculus', 'Periaqueductal gray']
updown_df = state_data[~state_data['region'].isin(up_areas)]

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
plot_order = state_data.sort_values(by='state_frequency', ascending=False).region.values
sns.pointplot(state_data, x='region', y='state_frequency', ax=axs, order=plot_order)
axs.set_xticklabels(all_regions, rotation=40)
sns.despine(trim=True)

# %% Load synchrony data

brunelX = pd.read_csv(os.path.join(analysed_data_dir, 'brunelX_S0_G[0, 1, 2, 3]RateAdj1.csv'))
colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)

sns.lineplot(brunelX, x='G', y='brunel_X')
axs.set_xticks([0,1,2,3])
axs.set_xticklabels(labels=[0,1,2,3])
axs.set_xlabel('Coupling strength (G)')
axs.set_ylabel('Synchrony (X)')
sns.despine(trim=True)
plt.savefig(os.path.join(figure_dir, f'network_synchrony_brunelX.pdf'), dpi=600, bbox_inches="tight")


# %% Plot all traces

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

plt.savefig(os.path.join(figure_dir, f'network_UP-DOWN_states_S0_G2.pdf'), dpi=600, bbox_inches="tight")


# %% serotonin trajectory

start_time = 0  # Start time in seconds
end_time = 3  # End time in seconds
step_size = 0.0002  # Step size in seconds
time = np.arange(start_time, end_time, step_size)
trajectory = np.array([0, 1, 0.7, 0.5, 0.45, 0.3, 0])
x_labels = np.linspace(start_time, end_time, len(trajectory))
print(x_labels)
# interpolate the trajectory values at the specified time points
interpolated_trajectory = np.interp(time, np.linspace(start_time, end_time, len(trajectory)), trajectory)

colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
axs.plot(interpolated_trajectory)
axs.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000])
axs.set_xticklabels(x_labels)
sns.despine(trim=True)
axs.set_xlabel('Time (s)')
axs.set_ylabel('Serotonin release')


plt.savefig(os.path.join(figure_dir, f'serotonin_trajectory.pdf'), dpi=600, bbox_inches="tight")

