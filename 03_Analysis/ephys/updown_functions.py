# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from scipy.stats import binned_statistic
from scipy.signal import convolve
from scipy.stats import binned_statistic_2d
from scipy.signal.windows import gaussian
import pathlib
from os.path import join, realpath, dirname, isfile
from matplotlib import colors as matplotlib_colors
import json
from pathlib import Path
from iblutil.numerical import ismember
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
from one.api import ONE


def init_one(local=False, open_one=False):
    """
    Initialize an instance of the ONE class with specified configuration.
    Parameters:
    local (bool): If True, initializes ONE in 'local' mode. Defaults to False, which sets the mode to 'auto'.
    open_one (bool): If True, initializes ONE with a specific base URL and credentials for the Open Alyx instance. Defaults to False.
    Returns:
    ONE: An instance of the ONE class configured based on the provided parameters.
    """
    if local:
        mode='local'
    else:
        mode='remote'
    if open_one:
        one = ONE(mode=mode, base_url='https://openalyx.internationalbrainlab.org',
                  password='international', silent=True)
    else:
        one = ONE(mode=mode)
    return one


def paths(save_dir='repo'):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input

    Save directory can be either the repository (for small files) or the one cache directory
    (for large files)

    Input
    ------------------------
    save_dir : str
        'repo' or 'cache' for saving in the repository or one cache, respectively

    Output
    ------------------------
    fig_path : str
        Path to where to save the figures

    save_path : str
        Path to where to save the output data
    """
    if not isfile(join(dirname(realpath(__file__)), '../paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['save_path'] = input('Path folder to data: ')
        path_file = open(join(dirname(realpath(__file__)), '../paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), '../paths.json')) as json_file:
        path_dict = json.load(json_file)
    if save_dir == 'cache':
        one = ONE(mode='local')
        save_dir = Path(one.cache_dir, 'serotonin')
        save_dir.mkdir(exist_ok=True)
    elif save_dir == 'repo':
        save_dir = path_dict['save_path']
    else:
        print('save_dir must be either repo or cache')
    return path_dict['fig_path'], save_dir


def load_subjects(only_expression=True):
    _, data_path = paths()
    subjects = pd.read_csv(join(data_path, 'subjects.csv'),
                           delimiter=';|,', engine='python')
    subjects['subject_nr'] = subjects['subject_nr'].astype(int)
    if only_expression:
        subjects = subjects[subjects['expression'] == 1]
    subjects = subjects.reset_index(drop=True)
    return subjects


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    subject_pal = sns.color_palette(sns.color_palette('Dark2'))
    frontal = sns.color_palette('Dark2')[1]
    sensory = sns.color_palette('Dark2')[5]
    hipp = sns.color_palette('Dark2')[4]
    amygdala = sns.color_palette('Dark2')[2]
    thalamus = sns.color_palette('Dark2')[0]
    striatum = sns.color_palette('Set1')[7]
    midbrain = sns.color_palette('Dark2')[3]

    colors = {'subject_palette': subject_pal,
              'grey': [0.7, 0.7, 0.7],
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.6, 0.6, 0.6],
              'awake': sns.color_palette('Dark2')[2],
              'anesthesia': sns.color_palette('Dark2')[3],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'stim': 'dodgerblue',
              'no-stim': [0.65, 0.65, 0.65],
              'NS': sns.color_palette('Set2')[0],
              'WS': sns.color_palette('Set2')[1],
              'WS1': sns.color_palette('Set2')[1],
              'WS2': sns.color_palette('Set2')[2],
              'main_states': sns.diverging_palette(20, 210, l=55, center='dark'),
              'Frontal cortex': frontal,
              'Sensory cortex': sensory,
              'Midbrain': midbrain,
              'Amygadala': amygdala,
              'Thalamus': thalamus,
              'Hippocampus': hipp,
              'Striatum': striatum,
              'OFC': sns.color_palette('Set1')[7],
              'mPFC': sns.color_palette('Dark2')[1],
              'M2': sns.color_palette('Dark2')[2],
              'Amyg.': sns.color_palette('Dark2')[6],
              'HPC': sns.color_palette('Dark2')[3],
              'VIS': sns.color_palette('Dark2')[5],
              'Pir.': sns.color_palette('Dark2')[4],
              'SC': sns.color_palette('Dark2')[7],
              'Thal.': sns.color_palette('tab10')[9],
              'PAG': sns.color_palette('Dark2')[0],
              'BC': sns.color_palette('Accent')[0],
              'Str.': sns.color_palette('Accent')[1],
              'MRN': sns.color_palette('Accent')[2],
              'OLF': sns.color_palette('tab10')[8],
              'Orbitofrontal cortex': sns.color_palette('Dark2')[0],
              'Medial prefrontal cortex': sns.color_palette('Dark2')[1],
              'Secondary motor cortex': sns.color_palette('Dark2')[2],
              'Amygdala': sns.color_palette('Dark2')[3],
              'Visual cortex': sns.color_palette('Dark2')[5],
              'Piriform cortex': sns.color_palette('Dark2')[6],
              'Superior colliculus': sns.color_palette('Dark2')[7],
              'Periaqueductal gray': sns.color_palette('Set1')[7],
              'Barrel cortex': sns.color_palette('Set2')[0],
              'Tail of the striatum': sns.color_palette('Set2')[1],
              'Midbrain reticular nucleus': sns.color_palette('Accent')[2],
              'Olfactory areas': sns.color_palette('tab10')[8],
              'Substantia nigra': [0.75, 0.75, 0.75],
              'Retrosplenial cortex': 'r',
              'RSP': 'r',
              'SNr': [0.75, 0.75, 0.75],
              'left-stim': sns.color_palette('Paired')[7],
              'left-no-stim': sns.color_palette('Paired')[6],
              'right-stim': sns.color_palette('Paired')[9],
              'right-no-stim': sns.color_palette('Paired')[8]}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


def add_significance(x, p_values, ax, y_pos='auto', alpha=0.05):
    """
    Add significance bars to a plot based on p-values.

    Parameters
    ----------
    x : array-like
        The x-coordinates corresponding to the p-values.
    p_values : array-like
        The p-values to evaluate for significance.
    ax : matplotlib.axes.Axes
        The axes object to plot the significance bars on.
    y_pos : float or 'auto', optional
        The y-coordinate for the significance bars. If 'auto', uses the top of the plot.
    alpha : float, optional
        The significance threshold (default: 0.05).
    """

    p_sig = p_values < alpha
    start_end = np.where(np.concatenate(([0], np.diff(p_sig).astype(int))))[0]
    if p_sig[0] == True:
        start_end = np.concatenate(([0], start_end))
    if p_sig[-1] == True:
        start_end = np.concatenate((start_end, [p_sig.shape[0]-1]))
    if y_pos == 'auto':
        y = ax.get_ylim()[1]
    else:
        y = y_pos
    for (i, ind) in zip(np.arange(start_end.shape[0])[::2], start_end[::2]):
        ax.plot([x[ind], x[start_end[i+1]]], [y + (y*0.05), y + (y*0.05)], color='k', lw=1.5,
                clip_on=False)


def get_artifact_neurons():
    artifact_neurons = pd.read_csv(
        join(pathlib.Path(__file__).parent.resolve(), 'artifact_neurons.csv'))
    return artifact_neurons


def remove_artifact_neurons(df):
    _, data_path = paths()
    artifact_neurons = pd.read_csv(join(data_path, 'artifact_neurons.csv'))
    for i, column in enumerate(df.columns):
        if df[column].dtype == bool:
            df[column] = df[column].astype('boolean')
    if 'pid' in df.columns:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['pid', 'neuron_id', 'subject', 'probe', 'date']).query('_merge=="left_only"').drop('_merge', axis=1)
    else:
        df = pd.merge(df, artifact_neurons, indicator=True, how='outer',
                      on=['subject', 'probe', 'date', 'neuron_id']).query('_merge=="left_only"').drop('_merge', axis=1)
    return df


def query_ephys_sessions(only_expression=True, acronym=None, one=None):
    """
    Query ephys recordings from the database.

    Parameters
    ----------
    only_expression : boolean
        Only return sessions from mice with expression
    acronym : string, optional
        Only return recordings that include the brain region with this Allen acronym
    one : Initialized connection to the ONE database

    Returns
    -------
    rec : DataFrame
        A dataframe with the identifiers of all ephys recordings.

    """
    if one is None:
        one = init_one()

    # Construct django query string
    DJANGO_STR = ('session__projects__name__icontains,serotonin_inference,'
                  'session__qc__lt,50,json__extended_qc__alignment_count__gt,0')

    # Query sessions
    if acronym is None:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR)
    elif type(acronym) is str:
        ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=acronym)
    else:
        ins = []
        for i, ac in enumerate(acronym):
            ins = ins + one.alyx.rest('insertions', 'list', django=DJANGO_STR, atlas_acronym=ac)

    # Only include subjects from subjects.csv
    incl_subjects = load_subjects(only_expression=only_expression)
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]
    
    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    rec = rec.drop_duplicates('pid', ignore_index=True)
    rec = rec.merge(incl_subjects[['subject', 'expression', 'also_awake']], on='subject', how='left')
    return rec


def remap(acronyms, source='Allen', dest='Beryl', combine=False, split_thalamus=False,
          abbreviate=True, brainregions=None):
    """
    Remap a list of brain region acronyms from one mapping source to another.
    Parameters:
        acronyms (list or array-like): A list of brain region acronyms to be remapped.
        source (str, optional): The source mapping to use for remapping. Default is 'Allen'.
        dest (str, optional): The destination mapping to remap to. Default is 'Beryl'.
        combine (bool, optional): If True, combines remapped regions into broader categories.
                                    Default is False.
        split_thalamus (bool, optional): If True and `combine` is True, splits thalamus regions
                                            into subcategories. Default is False.
        abbreviate (bool, optional): If True and `combine` is True, abbreviates combined region names.
                                        Default is True.
        brainregions (BrainRegions, optional): An instance of the BrainRegions class to use for
                                                remapping. If None, a new instance is created.
                                                Default is None.
    Returns:
        list or array-like: The remapped acronyms. If `combine` is True, returns combined regions
                            based on the specified options.
    """

    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    if combine:
        return combine_regions(remapped_acronyms, split_thalamus=split_thalamus, abbreviate=abbreviate)
    else:
        return remapped_acronyms


def combine_regions(acronyms, split_thalamus=False, abbreviate=True):
    """
    Combines regions into groups, input Beryl atlas acronyms: use remap function first
    """
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.in1d(acronyms, ['MOs'])] = 'M2'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'PO'
            regions[np.in1d(acronyms, ['LP'])] = 'LP'
            regions[np.in1d(acronyms, ['LD'])] = 'LD'
            regions[np.in1d(acronyms, ['RT'])] = 'RT'
            regions[np.in1d(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal.'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.in1d(acronyms, ['ZI'])] = 'ZI'
        regions[np.in1d(acronyms, ['PAG'])] = 'PAG'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'BC'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.in1d(acronyms, ['PIR'])] = 'Pir.'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'AI'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg.'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str.'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'HPC'
    else:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.in1d(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.in1d(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.in1d(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.in1d(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.in1d(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        regions[np.in1d(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.in1d(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.in1d(acronyms, ['AId', 'AIv', 'AIp'])] = 'Insular cortex'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        # regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.in1d(acronyms, ['PIR'])] = 'Piriform'
        # regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Striatum'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions


def high_level_regions(acronyms, merge_cortex=False, only_vis=True, input_atlas='Allen'):
    """
    Maps brain region acronyms to high-level brain regions based on the specified atlas and options.
    
    Parameters:
    -----------
    acronyms : list or array-like
        List of brain region acronyms to be mapped.
    merge_cortex : bool, optional
        If True, merges specific cortical regions into a single 'Cortex' category. 
        Default is False.
    only_vis : bool, optional
        If True, maps only visual cortex regions when `merge_cortex` is False. 
        Default is False.
    input_atlas : str, optional
        Specifies the input atlas to use for remapping. Default is 'Allen'.

    Returns:
    --------
    regions : numpy.ndarray
        Array of high-level brain region labels corresponding to the input acronyms.

    Notes:
    ------
    - The function uses predefined mappings to group acronyms into broader brain region categories.
    - If `merge_cortex` is True, regions like 'mPFC', 'OFC', 'M2', 'Pir', 'BC', and 'VIS' are grouped as 'Cortex'.
    - If `merge_cortex` is False and `only_vis` is True, only 'VIS' is mapped to 'Visual cortex'.
    - Specific mappings are applied for regions like 'Midbrain', 'Hippocampus', 'Thalamus', 'Amygdala', and 'Striatum'.
    """

    if input_atlas == 'Allen':
        acronyms = remap(acronyms)
    first_level_regions = combine_regions(acronyms, abbreviate=True)
    cosmos_regions = remap(acronyms, dest='Cosmos')
    regions = np.array(['root'] * len(first_level_regions), dtype=object)
    if merge_cortex:
        # regions[cosmos_regions == 'Isocortex'] = 'Cortex'
        # regions[first_level_regions == 'Pir'] = 'Cortex'
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2', 'Pir', 'BC', 'VIS'])] = 'Cortex'
    else:
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2'])] = 'Frontal cortex'
        regions[np.in1d(first_level_regions, ['Pir.'])] = 'Piriform cortex'
        if only_vis:
            regions[np.in1d(first_level_regions, ['VIS'])] = 'Visual cortex'
        else:
            regions[np.in1d(first_level_regions, ['Pir.', 'BC', 'VIS'])] = 'Sensory cortex'
    regions[cosmos_regions == 'MB'] = 'Midbrain'
    regions[cosmos_regions == 'HPF'] = 'Hippocampus'
    regions[cosmos_regions == 'TH'] = 'Thalamus'
    regions[np.in1d(first_level_regions, ['Amyg.'])] = 'Amygdala'
    regions[np.in1d(acronyms, ['CP', 'ACB', 'FS'])] = 'Striatum'
    return regions


def get_full_region_name(acronyms):
    """
    Retrieve the full region names corresponding to a list of brain region acronyms.
    This function takes a list of acronyms and attempts to map each acronym to its
    full region name using the BrainRegions class. If an acronym cannot be found,
    it is returned as-is. If the input contains only one acronym, the function
    returns a single string; otherwise, it returns a list of full region names.
    Args:
        acronyms (list of str): A list of brain region acronyms to be converted 
                                into full region names.
    Returns:
        str or list of str: The full region name corresponding to the acronym if 
                            the input is a single acronym, or a list of full region 
                            names if multiple acronyms are provided. If an acronym 
                            is not found, it is returned unchanged.
    """

    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def load_passive_opto_times(eid, one=None, freq=25):
    """
    Load in the time stamps of the optogenetic stimulation at the end of the recording, after the
    taks and the spontaneous activity. Or when it's a long stimulation session with different
    frequencies, only return those stimulation bouts of the specified Hz (default is 25 Hz).

    Returns
    opto_train_times : 1D array
        Timestamps of the start of each pulse train
    opto_pulse_times : 1D array
        Timestamps of all individual pulses
    """

    if one is None:
        one = init_one()
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['date']

    # Load in pulses from disk if already extracted
    _, save_path = paths()
    save_path = join(save_path, 'OptoTimes')
    if isfile(join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy')):
        opto_train_times = np.load(join(save_path, f'{subject}_{date}_pulse_trains_{freq}hz.npy'))
        opto_on_times = np.load(join(save_path, f'{subject}_{date}_ind_pulses_{freq}hz.npy'))
        return opto_train_times, opto_on_times
    else:
        return None, None
   

def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    """
    Load and process trial data for a given experiment session.
    Parameters:
    -----------
    eid : str
        Experiment ID for the session to load.
    laser_stimulation : bool, optional
        If True, includes laser stimulation data in the trials. Default is False.
    invert_choice : bool, optional
        If True, inverts the choice values in the trials. Default is False.
    invert_stimside : bool, optional
        If True, inverts the stimulus side and signed contrast values. Default is False.
    one : ONE, optional
        An instance of the ONE API to use for data loading. If None, a new instance is created.
    Returns:
    --------
    pd.DataFrame or None
        A pandas DataFrame containing processed trial data with the following columns:
        - stimOn_times: Times when the stimulus was presented.
        - feedback_times: Times when feedback was given.
        - goCue_times: Times when the go cue was presented.
        - probabilityLeft: Probability of the stimulus appearing on the left.
        - contrastLeft: Contrast of the stimulus on the left.
        - contrastRight: Contrast of the stimulus on the right.
        - feedbackType: Feedback type (-1 for incorrect, 1 for correct).
        - choice: Choice made by the subject (-1 for left, 1 for right).
        - firstMovement_times: Times of the first movement.
        - signed_contrast: Signed contrast of the stimulus (positive for right, negative for left).
        - laser_stimulation: Laser stimulation data (if `laser_stimulation` is True).
        - laser_probability: Probability of laser stimulation (if `laser_stimulation` is True).
        - probe_trial: Indicator for probe trials (if `laser_stimulation` is True).
        - correct: Binary indicator for correct trials (1 for correct, 0 for incorrect).
        - right_choice: Binary indicator for rightward choices (1 for right, 0 for left).
        - stim_side: Stimulus side (-1 for left, 1 for right).
        - time_to_choice: Time from stimulus onset to choice.
        - reaction_times: Reaction times (time from stimulus onset to first movement, if available).
        Returns None if no trials are available for the given session.
    Notes:
    ------
    - If `laser_stimulation` is True and the laser probability dataset is unavailable, 
      the function estimates the laser probability based on signed contrast and stimulation data.
    - The `invert_choice` and `invert_stimside` parameters allow for flipping the choice and stimulus 
      side values, respectively, for specific experimental conditions.
    """

    one = one or ONE()

    data = one.load_object(eid, 'trials')
    data = {your_key: data[your_key] for your_key in [
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times']}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(
            eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(
                eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1

    trials['time_to_choice'] = trials['feedback_times'] - trials['stimOn_times']
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['stimOn_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials


def get_neuron_qc(pid, one=None, ba=None, force_rerun=False):
    """
    Compute or load neuron quality control (QC) metrics for a given probe insertion.
    Parameters:
    -----------
    pid : str
        The probe insertion ID.
    one : ONE, optional
        An instance of the ONE API for data access. If not provided, a new instance will be created.
    ba : BrainAtlas, optional
        An instance of the BrainAtlas class for anatomical alignment. If not provided, no alignment is performed.
    force_rerun : bool, optional
        If True, forces recalculation of QC metrics even if they are already saved on disk. Default is False.
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the neuron QC metrics.
    Notes:
    ------
    - If QC metrics are already computed and saved on disk, they will be loaded unless `force_rerun` is set to True.
    - QC metrics are saved to a CSV file in the session's ALF directory after computation.
    - The function uses spike sorting data to calculate QC metrics, which include spike times, cluster IDs, amplitudes, and depths.
    """
    from brainbox.metrics.single_units import spike_sorting_metrics    
    from brainbox.io.one import SpikeSortingLoader
    one = one or ONE()

    # Check if QC is already computed
    eid, probe = one.pid2eid(pid)
    session_path = one.eid2path(eid)
    if isfile(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv')) & ~force_rerun:
        print('Neuron QC metrics loaded from disk')
        qc_metrics = pd.read_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
        return qc_metrics

    # Load in spikes
    if ba is None:
        sl = SpikeSortingLoader(pid=pid, one=one)
    else:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Calculate QC metrics
    print('Calculating neuron QC metrics')
    qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters,
                                          spikes.amps, spikes.depths,
                                          cluster_ids=np.arange(clusters.channels.size))
    qc_metrics.to_csv(join(session_path, 'alf', probe, 'neuron_qc_metrics.csv'))
    return qc_metrics


def plot_scalar_on_slice(
        regions, values, coord=-1000, slice='coronal', mapping='Beryl', hemisphere='left',
        cmap='viridis', background='boundary', clevels=None, brain_atlas=None, colorbar=False, ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top' (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: AllenAtlas object
    :param colorbar: whether to plot a colorbar
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_cslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_sslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(
                coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_hslice(
                coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01,
                vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(
                volume='value', region_values=region_values, mapping=map, cmap=cmap,
                vmin=clevels[0], vmax=clevels[1], ax=ax)
            ba.plot_top(
                volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    return fig, ax


def make_bins(signal, timestamps, start_times, stop_times, binsize):
    """
    Bin a signal into specified time intervals and compute the mean value for each bin.
    Parameters:
        signal (array-like): The signal values to be binned.
        timestamps (array-like): The timestamps corresponding to the signal values.
        start_times (array-like): The start times of the intervals to bin the signal.
        stop_times (array-like): The stop times of the intervals to bin the signal.
        binsize (float): The size of each bin in seconds.
    Returns:
        list: A list of arrays, where each array contains the mean values of the signal
              for the bins within the corresponding interval defined by start_times and stop_times.
    """


    # Loop over start times
    binned_signal = []
    for (start, end) in np.vstack((start_times, stop_times)).T:
        binned_signal.append(binned_statistic(timestamps, signal, bins=int((end-start)*(1/binsize)),
                                              range=(start, end), statistic=np.nanmean)[0])
    return binned_signal


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """
    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :return: peths, binned_spikes
    :rtype: peths: Bunch({'mean': peth_means, 'std': peth_stds, 'tscale': ts, 'cscale': ids})
    :rtype: binned_spikes: np.array (n_align_times, n_clusters, n_bins)
    """

    # initialize containers
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes_[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
    return peths, binned_spikes


def binned_rate_timewarped(spike_times, spike_clusters, trials_df, start='stimOn_times',
        end='firstMovement_times', n_bins=10):
    """
    Compute time-warped binned firing rates for neurons across trials.
    This function calculates the firing rates of neurons by binning spike times
    within specified trial intervals. The intervals are defined by start and end
    times for each trial, and the spike times are warped to fit within these intervals.
    Parameters:
    -----------
    spike_times : array-like
        1D array of spike times (in seconds).
    spike_clusters : array-like
        1D array of cluster IDs corresponding to each spike time.
    trials_df : pandas.DataFrame
        DataFrame containing trial information. Must include columns specified
        by the `start` and `end` parameters.
    start : str, optional
        Column name in `trials_df` indicating the start times of trials. Default is 'stimOn_times'.
    end : str, optional
        Column name in `trials_df` indicating the end times of trials. Default is 'firstMovement_times'.
    n_bins : int, optional
        Number of bins to divide each trial interval into. Default is 10.
    Returns:
    --------
    binned_rate : numpy.ndarray
        3D array of shape (n_trials, n_neurons, n_bins) containing the firing rates
        of neurons in each bin for each trial. Firing rates are computed as spike
        counts divided by bin width.
    neuron_ids : numpy.ndarray
        1D array of unique neuron IDs corresponding to the `spike_clusters` input.
    Notes:
    ------
    - The function uses `np.digitize` to assign spikes to bins and `binned_statistic_2d`
        to count spikes per neuron per bin.
    - Spike times outside the trial interval are excluded from the computation.
    - The bin width is computed as the average width of the bins within each trial.
    """
    
    # Precompute unique neuron IDs and number of neurons
    neuron_ids = np.unique(spike_clusters)
    n_neurons = neuron_ids.shape[0]
    n_trials = trials_df.shape[0]

    # Initialize binned_rate array
    binned_rate = np.zeros((n_trials, n_neurons, n_bins))

    # Loop over trials
    for i, (start_time, end_time) in enumerate(zip(trials_df[start], trials_df[end])):

        # Define bin edges
        bin_edges = np.linspace(start_time, end_time, n_bins+1)
        bin_width = (end_time - start_time) / n_bins  # Compute bin width

        # Use digitize to assign each spike to a bin
        bin_indices = np.digitize(spike_times, bin_edges, right=True) - 1  # Subtract 1 to get 0-based index

        # Mask to keep only spikes within the trial interval
        valid_spike_mask = (spike_times >= start_time) & (spike_times <= end_time)

        # Filter spike data
        valid_clusters = spike_clusters[valid_spike_mask]
        valid_bins = bin_indices[valid_spike_mask]

        # Use binned_statistic_2d to count spikes per neuron per bin
        spike_counts, _, _, _ = binned_statistic_2d(
            valid_clusters, valid_bins, None, statistic='count',
            bins=[n_neurons, n_bins], range=[[0, n_neurons], [0, n_bins]]
        )

        # Convert to firing rate
        binned_rate[i, :, :] = spike_counts / bin_width

    return binned_rate, neuron_ids


def peri_multiple_events_time_histogram(
        spike_times, spike_clusters, events, event_ids, cluster_id,
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, error_bars='sem', ax=None,
        pethline_kwargs=[{'color': 'blue', 'lw': 2}, {'color': 'red', 'lw': 2}],
        errbar_kwargs=[{'color': 'blue', 'alpha': 0.5}, {'color': 'red', 'alpha': 0.5}],
        raster_kwargs=[{'color': 'blue', 'lw': 0.5}, {'color': 'red', 'lw': 0.5}],
        eventline_kwargs={'color': 'black', 'alpha': 0.5}, **kwargs):
    """
    Plot peri-event time histograms, with the meaning firing rate of units centered on a given
    series of events. Can optionally add a raster underneath the PETH plot of individual spike
    trains about the events.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds)
    spike_clusters : array-like
        Cluster identities for each element of spikes
    events : array-like
        Times to align the histogram(s) to
    event_ids : array-like
        Identities of events
    cluster_id : int
        Identity of the cluster for which to plot a PETH

    t_before : float, optional
        Time before event to plot (default: 0.2s)
    t_after : float, optional
        Time after event to plot (default: 0.5s)
    bin_size :float, optional
        Width of bin for histograms (default: 0.025s)
    smoothing : float, optional
        Sigma of gaussian smoothing to use in histograms. (default: 0.025s)
    as_rate : bool, optional
        Whether to use spike counts or rates in the plot (default: `True`, uses rates)
    include_raster : bool, optional
        Whether to put a raster below the PETH of individual spike trains (default: `False`)
    error_bars : {'std', 'sem', 'none'}, optional
        Defines which type of error bars to plot. Options are:
        -- `'std'` for 1 standard deviation
        -- `'sem'` for standard error of the mean
        -- `'none'` for only plotting the mean value
        (default: `'std'`)
    ax : matplotlib axes, optional
        If passed, the function will plot on the passed axes. Note: current
        behavior causes whatever was on the axes to be cleared before plotting!
        (default: `None`)
    pethline_kwargs : dict, optional
        Dict containing line properties to define PETH plot line. Default
        is a blue line with weight of 2. Needs to have color. See matplotlib plot documentation
        for more options.
        (default: `{'color': 'blue', 'lw': 2}`)
    errbar_kwargs : dict, optional
        Dict containing fill-between properties to define PETH error bars.
        Default is a blue fill with 50 percent opacity.. Needs to have color. See matplotlib
        fill_between documentation for more options.
        (default: `{'color': 'blue', 'alpha': 0.5}`)
    eventline_kwargs : dict, optional
        Dict containing fill-between properties to define line at event.
        Default is a black line with 50 percent opacity.. Needs to have color. See matplotlib
        vlines documentation for more options.
        (default: `{'color': 'black', 'alpha': 0.5}`)
    raster_kwargs : dict, optional
        Dict containing properties defining lines in the raster plot.
        Default is black lines with line width of 0.5. See matplotlib vlines for more options.
        (default: `{'color': 'black', 'lw': 0.5}`)

    Returns
    -------
        ax : matplotlib axes
            Axes with all of the plots requested.
    """
    from brainbox import singlecell

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')
    if not all(np.isfinite(events)):
        raise ValueError('There are NaN or inf values in the list of events passed. '
                         ' Please remove non-finite data points and try again.')

    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curves and add error bars
    mean_max, bars_max = [], []
    for i, event_id in enumerate(np.unique(event_ids)):
        # Compute peths
        peths, binned_spikes = singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                          events[event_ids == event_id], t_before,
                                                          t_after, bin_size, smoothing, as_rate)
        mean = peths.means[0, :]
        ax.plot(peths.tscale, mean, **pethline_kwargs[i])
        if error_bars == 'std':
            bars = peths.stds[0, :]
        elif error_bars == 'sem':
            bars = peths.stds[0, :] / np.sqrt(np.sum(event_ids == event_id))
        else:
            bars = np.zeros_like(mean)
        if error_bars != 'none':
            ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs[i])
        mean_max.append(mean.max())
        bars_max.append(bars[mean.argmax()])

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    plot_edge = (np.max(mean_max) + bars_max[np.argmax(mean_max)]) * 1.05
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        ax.axhline(0., color='black', lw=0.5)
        tickheight = plot_edge / len(events)  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        ii = 0
        for k, event_id in enumerate(np.unique(event_ids)):
            for i, t in enumerate(events[event_ids == event_id]):
                idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
                event_spks = clu_spks[idx]
                ax.vlines(event_spks - t, tickedges[i + ii + 1], tickedges[i + ii],
                          **raster_kwargs[k])
            ii += np.sum(event_ids == event_id)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return
