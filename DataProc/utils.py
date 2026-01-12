import os
import re
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import gridspec
def load_mask(mask_file):
    if mask_file and os.path.exists(mask_file):
        try:
            indices = []
            with open(mask_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.replace(',', ' ').split()
                    for p in parts:
                        if '-' in p:
                            start, end = map(int, p.split('-'))
                            indices.extend(range(start, end + 1))
                        else:
                            indices.append(int(p))
            return np.unique(np.array(indices, dtype=int))
        except Exception as e:
            print(f"警告: 无法加载掩膜文件 {mask_file}: {e}")
    return None


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
    ''' 将数据填充为 512 的倍数 (如果有必要)，
        假设输入为 (time, freq) 形状的 2D 数组。 '''
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
    消色散分发函数。

    默认调用 Numba 加速版本。如果 use_numba 为 False，
    它将调用纯 NumPy 版本以避免 JIT 编译开销。
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
    """内部纯 NumPy 版本。"""
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
    
    # 校正 peak_time 计算：
    # np.argmax(profile) 返回的是缩放后图片的时间索引 (范围 0-511)
    # 我们需要将其映射回真实的物理时长
    # 真实物理时长 = freq_reso * time_reso * tdownsamp (即 1:1 切片逻辑)
    # 图片时间轴长度 = w (通常为 512)
    # 所以：每个像素代表的时间 = (freq_reso * time_reso * tdownsamp) / w
    pixel_dt = (freq_reso * time_reso * tdownsamp) / w
    dpeak_time = np.argmax(profile) * pixel_dt
    
    # 绝对到达时间 = 块起始偏移量 + 块内峰值时间
    peak_time = offset + dpeak_time

    # block Start MJD
    start_mjd = tstart + offset / 86400.0
    burst_mjd = start_mjd + dpeak_time / 86400.0
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.scatter(np.argmax(profile), np.max(profile), color='red', s=100, marker='x')
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.scatter(np.argmax(profile), 0, color='red', s=100, marker='x')
    
    # Y轴: 根据图像高度(h)设置刻度位置，标签显示频率(MHz)
    # 之前代码中 f 计算逻辑有误，导致在非整倍数降采样时刻度位置错乱或未铺满
    plt.yticks(np.linspace(0, h, 6), np.linspace(freq.min(), freq.max(), 6).astype(int))
    
    # X轴: 根据图像宽度(w)设置刻度位置，标签显示时间(s)
    # 修正逻辑：由于我们在 main 中强制 1:1 切分 (block_len = freq_reso)，
    # 图片代表的总物理时间为 freq_reso * time_reso * tdownsamp，与最终 resize 后的宽度 w 无关。
    duration = freq_reso * time_reso * tdownsamp
    plt.xticks(np.linspace(0, w, 6), np.round(offset + np.linspace(0, duration, 6), 2))
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    # 更新文件名格式，包含 MJD
    # 增加跨平台文件命名安全性处理：替换 Windows/Linux 非法字符为连字符
    raw_name = f'{base_name}_S{start_mjd:.9f}_MJD{burst_mjd:.9f}_{peak_time:.4f}s'
    safe_name = re.sub(r'[<>:"/\\|?*]', '-', raw_name)
    output_basename = os.path.join(output_dir, safe_name)
    plt.savefig(f'{output_basename}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(f'{output_basename}.npy', data)                
    return None