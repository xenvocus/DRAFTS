import os
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import gridspec
def preprocess_data(data, exp_cut=5):
    """
    Preprocess data with normalization and clipping.
    
    Args:
        data: Input array of shape (H, W) for single image or (N, H, W) for batch
        exp_cut: Percentile cutoff for clipping
    
    Returns:
        Preprocessed array of the same shape as input
    """
    # Handle both single and batch inputs
    is_batch = data.ndim == 3
    if not is_batch:
        data = data[np.newaxis, ...]  # Add batch dimension
    
    # Ensure we're working with a copy to avoid modifying input
    data = data.copy()
    
    # Add offset
    data = data + 1
    
    # Normalize by mean along time axis (axis=1 for batch)
    # For batch (N, H, W), we normalize each sample independently
    data /= np.mean(data, axis=(1, 2), keepdims=True)
    
    # Compute percentiles and clip for each sample in batch
    vmin = np.nanpercentile(data, exp_cut, axis=(1, 2), keepdims=True)
    vmax = np.nanpercentile(data, 100-exp_cut, axis=(1, 2), keepdims=True)
    np.clip(data, vmin, vmax, out=data)
    
    # Min-max normalization per sample
    min_val = data.min(axis=(1, 2), keepdims=True)
    max_val = data.max(axis=(1, 2), keepdims=True)
    data -= min_val
    data /= (max_val - min_val + 1e-8)  # Add epsilon to avoid division by zero
    
    # Remove batch dimension if input was single
    if not is_batch:
        data = data[0]
    
    return data


def data_padding(data):
    ''' Pad data to be multiple of 512 in both dimensions, 
        assuming 2D array input with shape (time, freq). '''
    t, f = data.shape
    if f % 512:
        pad_width = 512 - (f % 512)
        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    if t % 512:
        pad_width = 512 - (t % 512)
        data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return data


def dedisperse(data, shifts, ds_chunk, use_numba=True):
    """
    Dispatcher function for dedispersion.

    Calls the Numba-accelerated version by default. If use_numba is False,
    it calls the pure NumPy version to avoid JIT compilation overhead.
    """
    if use_numba:
        return _dedisperse_numba(data, shifts, ds_chunk)
    else:
        return _dedisperse_numpy(data, shifts, ds_chunk)


@njit(parallel=True, fastmath=True, cache=True)
def _dedisperse_numba(data, shifts, ds_chunk):
    n_chan = data.shape[1]
    out = np.empty((ds_chunk, n_chan), dtype=np.float32)
    for j in prange(n_chan):
        s = shifts[j]
        out[:, j] = data[s:s + ds_chunk, j]
    return out


def _dedisperse_numpy(data, shifts, ds_chunk):
    """Internal pure NumPy version."""
    n_chan = data.shape[1]
    out = np.empty((ds_chunk, n_chan), dtype=np.float32)
    for j in range(n_chan):
        s = shifts[j]
        out[:, j] = data[s:s + ds_chunk, j]
    return out


def plot_burst(plot_datas, filename, offset, file_info, tdownsamp, output_dir):
    data, file_tstart = plot_datas
    base_name = os.path.basename(os.path.splitext(filename)[0])
    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)
    time_reso, freq_reso, tstart, _, freq = file_info
    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    peak_time    = offset + np.argmax(profile) * time_reso * tdownsamp
    all_time = peak_time + (file_tstart - tstart) * 86400
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.scatter(np.argmax(profile), np.max(profile), color='red', s=100, marker='x')
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])
    f = np.ceil(freq_reso/512)
    f = freq_reso/f
    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.scatter(np.argmax(profile), 0, color='red', s=100, marker='x')
    plt.yticks(np.linspace(0, f, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(np.linspace(0, w, 6), np.round(offset + np.arange(6)/5 * time_reso * tdownsamp * 512, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    output_basename = os.path.join(output_dir, f'{base_name}-{all_time:.4f}-{peak_time:.4f}')
    plt.savefig(f'{output_basename}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(f'{output_basename}.npy', data)                
    return None