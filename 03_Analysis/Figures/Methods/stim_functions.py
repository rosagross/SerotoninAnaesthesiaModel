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
import pathlib
from os.path import join, realpath, dirname, isfile
from glob import glob
from matplotlib import colors as matplotlib_colors
import json




# Number of states of HMM
N_STATES = 6


def load_subjects(anesthesia='all', behavior=None):
    assert anesthesia in ['no', 'yes', 'both', 'all', 'no&both', 'yes&both']
    subjects = pd.read_csv(join(pathlib.Path(__file__).parent.resolve(), 'subjects.csv'),
                           delimiter=';|,', engine='python')
    subjects = subjects[subjects['include'] == 1]
    if anesthesia == 'yes':
        subjects = subjects[subjects['anesthesia'] == 2]
    elif anesthesia == 'no':
        subjects = subjects[subjects['anesthesia'] == 0]
    elif anesthesia == 'both':
        subjects = subjects[subjects['anesthesia'] == 1]
    elif anesthesia == 'no&both':
        subjects = subjects[(subjects['anesthesia'] == 1) | (subjects['anesthesia'] == 0)]
    elif anesthesia == 'yes&both':
        subjects = subjects[(subjects['anesthesia'] == 1) | (subjects['anesthesia'] == 2)]
    subjects['subject_nr'] = subjects['subject_nr'].astype(int)
    subjects = subjects.reset_index(drop=True)
    return subjects


def paths():
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)
    return path_dict['fig_path'], path_dict['save_path']


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
    subject_pal = sns.color_palette(
        np.concatenate((sns.color_palette('tab20'),
                        [matplotlib_colors.to_rgb('maroon'), np.array([0, 0, 0])])))
    colors = {'subject_palette': subject_pal,
              'grey': [0.7, 0.7, 0.7],
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.7, 0.7, 0.7],
              'awake': sns.color_palette('Dark2')[2],
              'anesthesia': sns.color_palette('Dark2')[3],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'down-state': sns.color_palette('colorblind')[3],
              'up-state': [1, 1, 1],
              'stim': [0, 0, 0],
              'no-stim': [0.7, 0.7, 0.7],
              'NS': sns.color_palette('Set2')[0],
              'WS': sns.color_palette('Set2')[1],
              'WS1': sns.color_palette('Set2')[1],
              'WS2': sns.color_palette('Set2')[2],
              'states': 'Dark2',
              'states_light': 'Set2',
              'main_states': sns.diverging_palette(20, 210, l=55, center='dark'),
              'OFC': sns.color_palette('Dark2')[0],
              'mPFC': sns.color_palette('Dark2')[1],
              'M2': sns.color_palette('Dark2')[2],
              'Amyg': sns.color_palette('Dark2')[3],
              'Hipp': sns.color_palette('Dark2')[4],
              'VIS': sns.color_palette('Dark2')[5],
              'Pir': sns.color_palette('Dark2')[6],
              'SC': sns.color_palette('Dark2')[7],
              'Thal': sns.color_palette('tab10')[9],
              'PAG': sns.color_palette('Set1')[7],
              'BC': sns.color_palette('Accent')[0],
              'Str': sns.color_palette('Accent')[1],
              'MRN': sns.color_palette('Accent')[2],
              'OLF': sns.color_palette('tab10')[8],
              'RSP': 'r',
              'SNr': [0.75, 0.75, 0.75]}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi




def query_ephys_sessions(aligned=True, behavior_crit=False, n_trials=0, anesthesia='no',
                         acronym=None, one=None):
    assert anesthesia in ['no', 'both', 'yes', 'all', 'no&both', 'yes&both']
    if one is None:
        one = ONE()

    # Construct django query string
    DJANGO_STR = ('session__project__name__icontains,serotonin_inference,'
                 'session__qc__lt,50')
    if aligned:
        # Query all ephys-histology aligned sessions
        DJANGO_STR += ',json__extended_qc__alignment_count__gt,0'

    if behavior_crit:
        # Query sessions with an alignment and that meet behavior criterion
        DJANGO_STR += ',session__extended_qc__behavior,1'

    # Query sessions with at least this many trials
    if n_trials > 0:
        DJANGO_STR += f',session__n_trials__gte,{n_trials}'

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
    incl_subjects = load_subjects(anesthesia=anesthesia)
    ins = [i for i in ins if i['session_info']['subject'] in incl_subjects['subject'].values]

    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    rec = rec.drop_duplicates('pid', ignore_index=True)
    return rec

