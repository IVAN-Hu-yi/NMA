# Numerical and plotting packages
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import natsort
from scipy.signal import sosfiltfilt, butter, hilbert
# Libraries needed for this notebook to interact with the DANDI API
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import pynwb

# Libraries needed for this notebook to interact with NWB events
from ndx_events import LabeledEvents, AnnotatedEventsTable, Events

# FSSpec is a library that allows us to read files from the cloud
import fsspec

# NWB is based on HF5, so we need this library to read NWB files
import h5py
from fsspec.implementations.cached import CachingFileSystem
import matplotlib
import os
import nilearn as ni
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn.plotting import plot_connectome

def get_nwb(sbj, session):
    """ return nwbfile

    Parameters
    ----------
    sbj : int
        subject number
    session : int
        session number

    Returns
    -------
    _type_
        _description_
    """

    path = f'sub-{sbj}/sub-{sbj}_ses-{session}_behavior+ecephys.nwb'
    with DandiAPIClient() as client:
        asset = client.get_dandiset("000055").get_asset_by_path(
            path=path)
        s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
        

    # Note, caching is set once per access. If you want to change the cache location, you will need to restart the kernel.
    fs = CachingFileSystem(
        fs=fsspec.filesystem("http"),
        cache_storage="nwb-cache",  # Local folder for the cache
    )

    f = fs.open(s3_path, "rb")
    file = h5py.File(f)
    io = NWBHDF5IO(file=file, mode='r', load_namespaces=True)
    nwbfile = io.read()

    return nwbfile, fs

def extract_phys_accod_timestamps(phys_data:h5py._hl.dataset.Dataset, behavior_timestamps:pd.DataFrame, trange=list):

    """ Return a dictionary that contains EOG/ECG recordings according timestamps

    Args:
        phys_data: recordings from nwb files
        behavior_timestaps: a dataframe containing the timestamps labelled with specific activity
        trange: a list denoting the temporal region to look at, measured in hour

    """

    if len(trange) != 2:
        raise ValueError(f'trange should contains 2 elements instead of {len(trange)}')
    # Identify the start/end indices for each continuous chunk of the given behavioral label
    start_limit, end_limit = trange[0]*3600, trange[1]*3600 # convert to seconds
    len_timepoints = phys_data.shape[0] # get the number of recording timepoints
    phys_behavior = {}

    for i in behavior_timestamps['start_time'].index:

        start_time, end_time = behavior_timestamps['start_time'][i], behavior_timestamps['stop_time'][i]
        
        start_t = int((start_time/86400) * len_timepoints)
        end_t = int((end_time/86400) * len_timepoints) 

        if (start_time >= start_limit and start_time <= end_limit):
            if (end_time <= end_limit and end_time >= start_limit): # if within the range
                phys_behavior[str(start_time) + '_' + str(end_time)] = phys_data[start_t:end_t, :]
                
            if (end_time >= end_limit and end_time >= start_limit): # when end_time falls outside of the region

                phys_behavior[str(start_time) + '_' + str(end_limit)] = phys_data[start_t:int((end_limit/86400) * len_timepoints), :]
                

        elif (start_time <= start_limit and start_time <= end_limit):
            if (end_time <= end_limit and end_time >= start_limit):
                phys_behavior[str(start_limit) + '_' + str(end_limit)] = phys_data[int((start_limit/86400) * len_timepoints):end_t, :]
                
            if (end_time>=start_limit and end_time >= end_limit):
                phys_behavior[str(start_limit) + '_' + str(end_limit)] = phys_data[int((start_limit/86400) * len_timepoints):int((end_limit/86400) * len_timepoints), :]
                 
        else:
            pass
    
    return phys_behavior


def mni_2_csv(file:pynwb.file.NWBFile, sbj:str, session:str, retu = False):
    """Convert electrode locations from mni coods to csv file 

    Parameters
    ----------
    file : pynwb.file.NWBFile
        _description_
    sbj : str
        _description_
    session : str
        _description_
    """
    
    df = pd.DataFrame(file.electrodes.to_dataframe().iloc[:, 0:3])
    # df.iloc[:, :] = df.iloc[:, :] - 1.25 # map electrodes outside of area to the brain area
    df.to_csv(f'Data/coords/sbj_{sbj}_session_{session}_mni.csv', index=False, header=False)
    
    if retu:
        return df
    

def get_epochs(file:pynwb.file.NWBFile, behavior = False):
    """ Getting information about labels and epochs of the subject

    Parameters
    ----------
    file : pynwb.file.NWBFile
        An nwbfile containing ECOG recordings
    behaviour: Bool
        if `True`, store epochs for different behavioral type in a dict with key representing behaviour type and value as a dataframe of corresponding epochs

    Returns:
    ----------
    clabels_orig: pd.DataFrame
        Epochs of all behaviour
    blocklist: list
        interruption types
    acts: list
        type of behaviour
    epochs: dict, if behaviour=True
        behavior-specific epochs
    """

    clabels_orig = file.intervals['epochs'].to_dataframe()
    blocklist = []
    acts = []
    if behavior: # check if behaviour-specific epochs required
            epochs = {}

    # types of behavior
    all_acts = np.unique(clabels_orig['labels'])
    all_acts = all_acts[all_acts!=''] # remove empty string

    # subset blocklist and current action
    for _, act in enumerate(all_acts):
        if act.startswith('Blocklist'):
            blocklist.append(act)
        else:
            acts.append(act)

        if behavior: # check if behaviour-specific epochs required
            epochs[act] = clabels_orig[clabels_orig['labels']==act]
        
    if behavior:
        return clabels_orig, blocklist, acts, epochs
    else:
        return clabels_orig, blocklist, acts

def plot_electrodes(coords:pd.DataFrame, sbj:str, figsize=(10, 5), dpi=300):


    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 1, height_ratios=[40, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    color_map = sns.color_palette("vlag", n_colors=coords.shape[0])
    im = plot_connectome(np.eye(coords.shape[0]), coords, title=f'Subject {sbj}', node_color=color_map, axes=ax0)
    ax1.imshow([np.arange(coords.shape[0]) for _ in range(2)], cmap="vlag")
    ax1.set_yticks([])
    plt.tight_layout()
    
    return fig

def epoching(data, rate=500, epoch=5, max_t = 90):
    """divide orignial data into epochs with a temporal window

    Parameters
    ----------
    data : ndarray
        relavant behavior data
    rate : int, optional
        sampling rate, by default 500
    epochl : int, optional
        epoch length, by default 5
    max_t : int, optional
        maximum length for each trial, by default 90 seconds

    Returns
    -------
    array: ndarray 
        with shape (epochLength, Number of Channel, Number of epochs)
    """

    step = epoch*rate
    max_idx = max_t*rate

    # epoch data array with epochLength*channel*No.Epoch
    array = np.zeros((step, data.shape[1], int(max_idx/step))) 

    for i, j in zip(range(0, max_idx, step), range(0, array.shape[2])):
        array[:, :, j] = data[i:i+step, :]
    
    return array


def extract_all_recordings(data:h5py._hl.dataset.Dataset, behavior:pd.DataFrame):

    """Return a list of data for a specific behavior over the dataset

    Parameters
    ----------
    data : h5py._hl.dataset.Dataset
        relavant behavior data
    behavior : pd.DataFrame
        dataframe with intervals about behaviors

    Returns
    -------
    _type_
        _description_
    """

    nrow = behavior.shape[0]
    behavior['start_idx'], behavior['stop_idx'] = behavior['start_time']/86400 * 43200000, behavior['stop_time']/86400 * 43200000

    
    datalist = []
    for i in range(nrow):
        start, stop = int(behavior.iloc[i, :]['start_idx']), int(behavior.iloc[i, :]['stop_idx'])
        datalist.append(data[start:stop, :])
    return datalist
    