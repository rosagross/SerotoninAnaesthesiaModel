#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 12:45:00 2026
By: Guido Meijer & Antigravity
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from os.path import join
from updown_functions import paths, figure_style, load_subjects, remap

# Config
MIN_NEURONS = 5
RECEPTORS = ['5-HT1a', '5-HT1b', '5-HT1f', '5-HT2a', '5-HT2c', '5-HT3a', '5-HT5a']

# Paths
fig_path, data_path = paths()
colors, dpi = figure_style()

# -----------------------------------------------------------------------------
# 1. Load and process receptor expression
# -----------------------------------------------------------------------------
expr_df = pd.read_csv(join(data_path, 'receptor_expression.csv'))
expr_df = expr_df[~np.isin(expr_df['acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo'])]
expr_df['region'] = remap(expr_df['acronym'])

# Filter for the target receptors
expr_df = expr_df[expr_df['receptor'].isin(RECEPTORS)]

# Get expression value per region per receptor (median expression energy)
expression_mean = expr_df[['region', 'receptor', 'expression_energy']].groupby(
    ['region', 'receptor']).median().reset_index()

# Pivot so that each receptor is a separate column
expression_pivot = expression_mean.pivot(
    index='region', columns='receptor', values='expression_energy'
).reset_index()

# -----------------------------------------------------------------------------
# 2. Load and process DR projection strength
# -----------------------------------------------------------------------------
proj_df = pd.read_csv(join(data_path, 'dr_projection_strength.csv'))
proj_df = proj_df[~np.isin(proj_df['allen_acronym'], ['MMme', 'CUL4, 5', 'CUL4, 5gr', 'CUL4, 5mo', 'onl',
                                                      'MMd', 'MMl', 'MMm', 'MMp'])]
proj_df['region'] = remap(proj_df['allen_acronym'])

# Get mean projection density per region
projection_mean = proj_df[['region', 'projection_density']].groupby(
    ['region']).mean().reset_index()

# -----------------------------------------------------------------------------
# 3. Load and process neural (ephys) data
# -----------------------------------------------------------------------------
ephys_data = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()
for nickname in np.unique(subjects['subject']):
    ephys_data.loc[ephys_data['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]

# Calculate summary per region (mean mod_index for expression & modulated neurons)
ephys_summary = ephys_data[(ephys_data['expression'] == 1) & (ephys_data['modulated'] == 1)][[
    'region', 'mod_index']].groupby('region').mean()['mod_index'].to_frame()
ephys_summary['n_neurons'] = ephys_data[(ephys_data['expression'] == 1) & (ephys_data['modulated'] == 1)].groupby(
    'region').size()
ephys_summary = ephys_summary.reset_index()

# Drop regions with too few neurons
ephys_summary = ephys_summary[ephys_summary['n_neurons'] >= MIN_NEURONS]

# -----------------------------------------------------------------------------
# 4. Merge datasets and prepare for GLM
# -----------------------------------------------------------------------------
merged_df = pd.merge(ephys_summary, projection_mean, on='region')
merged_df = pd.merge(merged_df, expression_pivot, on='region')

# Predictors: projection density + receptor expression levels
predictors = ['projection_density'] + RECEPTORS
analysis_df = merged_df[['region', 'mod_index'] + predictors].dropna().reset_index(drop=True)

# Standardize variables to get standardized beta coefficients
analysis_df_std = analysis_df.copy()
for col in predictors:
    analysis_df_std[col] = (analysis_df[col] - analysis_df[col].mean()) / analysis_df[col].std()
analysis_df_std['mod_index'] = (analysis_df['mod_index'] - analysis_df['mod_index'].mean()) / analysis_df['mod_index'].std()

# -----------------------------------------------------------------------------
# 5. Fit GLM
# -----------------------------------------------------------------------------
X = sm.add_constant(analysis_df_std[predictors])
y = analysis_df_std['mod_index']

# Gaussian GLM is equivalent to multiple linear regression
model = sm.GLM(y, X, family=sm.families.Gaussian())
results = model.fit()

print("\n=== GLM Fit Results (Predicting Modulation Index) ===")
print(results.summary())

# Extract coefficients and p-values
beta_weights = results.params[predictors]
p_values = results.pvalues[predictors]

# -----------------------------------------------------------------------------
# 6. Plotting
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(2.75, 2.5), dpi=dpi)

# Map internal column names to publication-friendly labels
plot_labels = ['DRN projection'] + RECEPTORS
bar_colors = [colors['sert']] + [colors['wt']] * len(RECEPTORS)

# Plot the bars
x_pos = np.arange(len(plot_labels))
bars = ax.bar(x_pos, beta_weights.values, color=bar_colors, edgecolor='none', width=0.6)

# Add reference line at zero
ax.axhline(0, color='grey', lw=0.5, ls='--')

# Annotate with significance stars
for i, bar in enumerate(bars):
    p = p_values.values[i]
    val = beta_weights.values[i]
    if p < 0.05:
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        else:
            stars = '*'
        
        # Position the star slightly above or below the bar based on the sign
        if val >= 0:
            y_text = val + 0.03
            va = 'bottom'
        else:
            y_text = val - 0.03
            va = 'top'
            
        ax.text(bar.get_x() + bar.get_width()/2., y_text, stars,
                ha='center', va=va, fontsize=8, color='k')

# Style the axes
ax.set_ylabel('Beta weight (std.)', fontsize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=7)
ax.set_title('Predictors of modulation', fontsize=7)

# Adjust y limits to leave space for stars
ymin, ymax = ax.get_ylim()
y_range = ymax - ymin
ax.set_ylim(ymin - 0.15 * y_range, ymax + 0.15 * y_range)

sns.despine(trim=True)
plt.tight_layout()

# Save figures
plt.savefig(join(fig_path, 'receptor_projection_glm.pdf'))
plt.savefig(join(fig_path, 'receptor_projection_glm.png'))
print(f"\nSaved plots to:\n- {join(fig_path, 'receptor_projection_glm.pdf')}\n- {join(fig_path, 'receptor_projection_glm.png')}")
plt.show()
