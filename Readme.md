# Serotonin induces DOWN states across the anesthetized mouse brain

📄 Data and code accompanying the [bioRxiv preprint](https://doi.org/10.64898/2026.07.14.738517)

## Dataset

This openly available dataset contains data from 38 Neuropixel insertions (typically two simultaneously) obtained from 7 lightly anesthetized mice. The dataset includes spiking activity of 2,422 single well-isolated neurons across eight brain regions: hippocampus, thalamus, amygdala, striatum, midbrain, and frontal, piriform & visual cortex. During these recordings, serotonergic neurons in the dorsal raphe nucleus were optogenetically stimulated at 1, 5, 10, and 25 Hz for 1 second with a 5-15s inter-stimulation interval.

The data is accessible via the [ONE API](https://int-brain-lab.github.io/ONE/one_reference.html). To install the API and other loading function do `pip install ibllib` after which the data can be queried using the following example code:
```
from one.api import ONE

# Initialize ONE API
one = ONE(mode='remote', base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)

# Query recordings
subjects = ['ZFM-04812', 'ZFM-04820', 'ZFM-05169', 'ZFM-05181', 'ZFM-05182', 'ZFM-05492', 'ZFM-05495']
subject_pattern = '^(' + '|'.join(subjects) + ')$'
ins = one.alyx.rest('insertions', 'list', django=
                    (f"session__subject__nickname__regex,{subject_pattern}",
                     'session__projects__name__icontains,serotonin_inference',
                     'session__qc__lt,50,json__extended_qc__alignment_count__gt,0'))

# Create a dataframe with the recording info
rec = pd.DataFrame()
rec['pid'] = np.array([i['id'] for i in ins])
rec['eid'] = np.array([i['session'] for i in ins])
rec['probe'] = np.array([i['name'] for i in ins])
rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
```

To load in the neural data of an example recording session use the following example code:
```
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()

# Load in the spiking data of the first recording
sl = SpikeSortingLoader(pid=rec['pid'].values[0], one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
```

The times and frequencies of optogenetic stimulation are not accessible via the API but can be found in the repository [here](https://github.com/rosagross/SerotoninAnaesthesiaModel/tree/main/02_Data/ephys/OptoTimes).
