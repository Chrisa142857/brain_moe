import os
import shutil
import glob
# import mne.io as io
import numpy as np
from mne.io import read_raw_eeglab
from tqdm import tqdm

def sample_eeg_data(filepath, target_shape=(22, 4, 200)):
    """
    Sample EEG data to target dimensions: channels x time_segments x points_per_patch
    
    Parameters:
    -----------
    filepath : str
        Path to the EEGLAB .set file
    target_shape : tuple
        Target dimensions (num_channels, time_segments, points_per_patch)
    
    Returns:
    --------
    sampled_data : numpy.ndarray
        Sampled EEG data with shape (22, 4, 200)
    """
    num_channels, time_segments, points_per_patch = target_shape
    
    # Load EEGLAB data
    # print("Loading EEG data...")
    raw = read_raw_eeglab(filepath, preload=True)
    
    # Get original sampling rate and data
    sfreq = raw.info['sfreq']  # Should be 250 Hz
    data = raw.get_data()  # Shape: (channels, time_points)
    
    # print(f"Original shape: {data.shape}")
    # print(f"Sampling rate: {sfreq} Hz")
    # print(f"Duration: {data.shape[1]/sfreq:.1f} seconds")
    
    # Step 1: Select channels
    # Option A: Select first 22 channels
    selected_channels = data[:num_channels, :]
    
    # Option B: Select specific channels (uncomment if you have channel preferences)
    # channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # selected_channels = data[channel_indices, :]
    
    # print(f"Selected channels shape: {selected_channels.shape}")
    
    # Step 2: Create time segments
    total_points_needed = time_segments * points_per_patch  # 4 * 200 = 800 points
    duration_needed = total_points_needed / sfreq  # 800/250 = 3.2 seconds
    
    # print(f"Total points needed: {total_points_needed}")
    # print(f"Duration needed: {duration_needed:.2f} seconds")
    
    # Extract data from the beginning (you can modify start_idx for different time periods)
    start_idx = 0
    end_idx = start_idx + total_points_needed
    
    if end_idx > selected_channels.shape[1]:
        raise ValueError(f"Not enough data points. Need {total_points_needed}, have {selected_channels.shape[1]}")
    
    # Extract the required time window
    windowed_data = selected_channels[:, start_idx:end_idx]
    
    # Step 3: Reshape into segments
    # Reshape to (channels, time_segments, points_per_patch)
    sampled_data = windowed_data.reshape(num_channels, time_segments, points_per_patch)
    
    # print(f"Final shape: {sampled_data.shape}")
    # 
    return sampled_data

root = '/ram/USERS/ziquanw/data/NATVIEW_EEGFMRI/projects/EEG_FMRI/data_indi_preproc/'
saveroot = 'data/fmrieeg-Shaefer_400'
fmri_task2fn = {}

for f in glob.glob('/ram/USERS/ziquanw/data/NATVIEW_EEGFMRI/projects/EEG_FMRI/data_indi_preproc/*/*/func/*/func_atlas/*400parcels7networks_desc-sm0_bold.tsv'):
    items = f.split('/')[-1].split('_')[2:4]
    if 'run' not in items[-1]: items = items[:-1]
    label = '_'.join(items)
    if label not in fmri_task2fn: fmri_task2fn[label] = []
    fmri_task2fn[label].append(f)

eeg_task2fn = {}
for f in glob.glob('/ram/USERS/ziquanw/data/NATVIEW_EEGFMRI/projects/EEG_FMRI/data_indi_preproc/*/*/eeg/*eeg.set'):
    items = f.split('/')[-1].split('_')[2:4]
    if 'run' not in items[-1]: items = items[:-1]
    label = '_'.join(items)
    if label not in eeg_task2fn: eeg_task2fn[label] = []
    eeg_task2fn[label].append(f)

# print(eeg_task2fn)
for label in fmri_task2fn:
    for f in tqdm(fmri_task2fn[label], desc=label):
        fn = f.split("/")[-1]
        items = fn.split('_')[:4]
        if 'run' not in items[-1]: items = items[:-1]
        newfn = '_'.join(items)
        eegfn = f'{newfn}_eeg.set'
        neweegfn = f'{newfn}_eeg.npy'
        newfn = f'{newfn}_bold.tsv'
        eegfns = [f.split('/')[-1] for f in eeg_task2fn[label]]
        if eegfn not in eegfns: continue
        # assert eegfn in eegfns, f'{eegfn} {eegfns}'
        eeg = eeg_task2fn[label][eegfns.index(eegfn)]
        eeg = sample_eeg_data(eeg)
        print(f, f'{saveroot}/BOLD/{newfn}', neweegfn)
        shutil.copy(f, f'{saveroot}/BOLD/{newfn}')
        np.save(f'{saveroot}/EEG/{neweegfn}', eeg)


