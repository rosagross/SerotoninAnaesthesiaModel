# %%
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from stim_functions import figure_style

matrix_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, os.pardir))
atlas_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), os.pardir, os.pardir, 'atlas_data'))
figure_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "Methods"))

# load data 
grouped_connectivity_df = pd.read_csv(os.path.join(matrix_dir, "MODEL_Cmatrix_grouped_cre-False_hemi-3_grouping-median_thresh-0.005.csv"))
grouped_connectivity_df.drop(['Unnamed: 0'], inplace=True, axis=1)
drn_connectivity = pd.read_csv(os.path.join(matrix_dir, "drn_connectivity_cre-True_hemi-3_grouping-median_thresh-0.005.csv"))
#drn_connectivity = np.array(drn_connectivity)
relevant_areas = pd.read_csv(os.path.join(atlas_dir, 'relevant_areas.csv'))
relevant_areas.drop(['Unnamed: 0'], inplace=True, axis=1)
relevant_areas = np.array(relevant_areas).flatten()

# abbreviations
region_abrv = ['Amyg', 'Hipp', 'mPFC', 'MRN', 'OLF', 'OFC', 'PAG', 'Pir', 'RSC', 'M2', 'SC', 'Str', 'Thal', 'Vis']

# %% plot connectivity matrix
colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75))

sns.heatmap(grouped_connectivity_df, xticklabels=region_abrv, yticklabels=region_abrv, cmap='flare_r',
            rasterized=True, square=True, ax=axs, cbar_kws={'label': 'Projection density'})
plt.title("Global connectivity")
plt.savefig(os.path.join(figure_dir, f'connectivity_matrix.pdf'), bbox_inches="tight")
plt.show()


# %% plot drn matrix
#colors, dpi = figure_style()
f, axs = plt.subplots(1, 1, figsize=(2, 1.75))
sns.heatmap(drn_connectivity, yticklabels=region_abrv, square=True, rasterized=True, cmap='flare_r', ax=axs, cbar_kws={'label': 'Projection density'})
plt.title("DRN connectivity")
axs.get_xaxis().set_visible(False)
plt.savefig(os.path.join(figure_dir, f'drn_connectivity.pdf'), bbox_inches="tight")
plt.show()


# %%
