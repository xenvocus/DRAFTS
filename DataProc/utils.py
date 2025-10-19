import os
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import gridspec
def preprocess_data(data, exp_cut=5):
    data = data + 1
    data /= np.mean(data, axis=0)
    vmin, vmax = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    np.clip(data, vmin, vmax, out=data)
    min_val = data.min()
    max_val = data.max()
    data -= min_val
    data /= (max_val - min_val)
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