import os
import re
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
def load_mask(mask_file):
    if mask_file and os.path.exists(mask_file):
        try:
            indices = []
            with open(mask_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'frequency range' in line.lower() and 'channel range' in line.lower():
                        continue

                    if '|' in line:
                        right = line.split('|', 1)[1]
                        nums = re.findall(r'\d+', right)
                        if len(nums) >= 2:
                            start, end = int(nums[0]), int(nums[1])
                            if start <= end:
                                indices.extend(range(start, end + 1))
                            else:
                                indices.extend(range(end, start + 1))
                            continue
                        if len(nums) == 1:
                            indices.append(int(nums[0]))
                            continue

                    parts = line.replace(',', ' ').split()
                    for p in parts:
                        if '-' in p:
                            start, end = map(int, p.split('-'))
                            if start <= end:
                                indices.extend(range(start, end + 1))
                            else:
                                indices.extend(range(end, start + 1))
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


def plot_burst(plot_datas, filename, offset, file_info, tdownsamp, output_dir, bbox=None, mask_idc=None, gradcam=None):
    data, file_tstart = plot_datas
    base_name = os.path.basename(os.path.splitext(filename)[0])
    
    time_reso, freq_reso, tstart, _, freq = file_info
    w, h = data.shape
    profile = np.mean(data, axis=1)

    # 布局调整：如果有 gradcam，则宽度加倍，显示左右两个图
    if gradcam is not None:
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(4, 2) # 4行2列
    else:
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(4, 1)

    # 校正 peak_time 计算：
    # np.argmax(profile) 返回的是缩放后图片的时间索引 (范围 0-511)
    # ... (省略中间注释)
    pixel_dt = (freq_reso * time_reso * tdownsamp) / w
    dpeak_time = np.argmax(profile) * pixel_dt
    
    # 绝对到达时间 = 块起始偏移量 + 块内峰值时间
    peak_time = offset + dpeak_time

    # block Start MJD
    start_mjd = tstart + offset / 86400.0
    burst_mjd = start_mjd + dpeak_time / 86400.0
    
    plt.subplots_adjust(wspace=0.1, hspace=0)

    # --- 左侧（或唯一）: 原始数据 ---
    # Profile
    ax_prof = plt.subplot(gs[0, 0])
    ax_prof.plot(profile, color='royalblue', alpha=0.8, lw=1)
    ax_prof.scatter(np.argmax(profile), np.max(profile), color='red', s=100, marker='x')
    ax_prof.set_xlim(0, w)
    ax_prof.set_xticks([])
    ax_prof.set_yticks([])
    
    # Dynamic Spectrum
    ax_ds = plt.subplot(gs[1:, 0])
    ax_ds.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    
    # 增加：标注被掩膜的通道 (红色短横线)
    if mask_idc is not None and len(mask_idc) > 0:
        dash_len = w * 0.03
        ax_ds.hlines(mask_idc, 0, dash_len, colors='red', linewidths=0.6, alpha=0.8)

    ax_ds.scatter(np.argmax(profile), 0, color='red', s=100, marker='x')
    
    # Y轴刻度
    ax_ds.set_yticks(np.linspace(0, h, 6))
    ax_ds.set_yticklabels(np.linspace(freq.min(), freq.max(), 6).astype(int))
    ax_ds.set_ylabel('Frequency (MHz)')
    
    # X轴刻度
    duration = freq_reso * time_reso * tdownsamp
    ax_ds.set_xticks(np.linspace(0, w, 6))
    ax_ds.set_xticklabels(np.round(offset + np.linspace(0, duration, 6), 2))
    ax_ds.set_xlabel('Time (s)')

    if bbox is not None:
        x_min, x_max, y_min, y_max = bbox
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=0.8, edgecolor='red', facecolor='none')
        ax_ds.add_patch(rect)

    # --- 右侧: Grad-CAM ---
    if gradcam is not None:
        # Profile (复制一份在右边，方便对比)
        ax_prof_cam = plt.subplot(gs[0, 1])
        ax_prof_cam.plot(profile, color='royalblue', alpha=0.8, lw=1)
        ax_prof_cam.set_xlim(0, w)
        ax_prof_cam.set_xticks([])
        ax_prof_cam.set_yticks([]) # 右侧不显示Y轴刻度
        ax_prof_cam.set_title("Grad-CAM Activation", fontsize=10)

        # Heatmap
        ax_cam = plt.subplot(gs[1:, 1])
        # 显示原始数据作为背景（灰度），叠加显眼的热力图
        ax_cam.imshow(data.T, origin='lower', cmap='gray', alpha=0.5, aspect='auto')
        im_cam = ax_cam.imshow(gradcam.T, origin='lower', cmap='jet', alpha=0.6, aspect='auto')
        
        # 同样的 bbox
        if bbox is not None:
            rect2 = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=0.8, edgecolor='white', linestyle='--', facecolor='none')
            ax_cam.add_patch(rect2)
            
        # 设置刻度 (与左图对齐但不显示Y轴标签)
        ax_cam.set_yticks(np.linspace(0, h, 6))
        ax_cam.set_yticklabels([]) # 隐藏Y轴标签
        ax_cam.set_xticks(np.linspace(0, w, 6))
        ax_cam.set_xticklabels(np.round(offset + np.linspace(0, duration, 6), 2))
        ax_cam.set_xlabel('Time (s)')


    # 更新文件名格式，包含 MJD
    # 增加跨平台文件命名安全性处理：替换 Windows/Linux 非法字符为连字符
    raw_name = f'{base_name}_S{start_mjd:.9f}_MJD{burst_mjd:.9f}_{peak_time:.4f}s'
    safe_name = re.sub(r'[<>:"/\\|?*]', '-', raw_name)
    output_basename = os.path.join(output_dir, safe_name)
    plt.savefig(f'{output_basename}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(f'{output_basename}.npy', data)                
    return None