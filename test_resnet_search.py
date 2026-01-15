import os

# 设置缓存目录到本地可写文件夹
if 'NUMBA_CACHE_DIR' not in os.environ:
    numba_cache_dir = os.path.join(os.getcwd(), 'numba_cache')
    os.makedirs(numba_cache_dir, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir

import torch
import warnings
import argparse
import numpy as np
import seaborn as sns
import onnxruntime as ort 
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
from datetime import datetime
from braceexpand import braceexpand
from DataProc import DataLoader, preload_worker
from concurrent.futures import ProcessPoolExecutor
from DataProc.utils import preprocess_data, dedisperse, plot_burst, data_padding, load_mask

strt_time = datetime.now()
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_color_codes()
block_size = 512
tdownsamp = 4
base_model = 'resnet18'
model_path = './class_resnet18.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-dm', '--dm', type=int, default=893)
    args.add_argument('-i', '--input', type=str, default='./')
    args.add_argument('-o', '--output', type=str, default='./')
    args.add_argument('-re', type=str, default='*.fits')
    args.add_argument('-p', '--prob', type=float, default=0.5)
    args.add_argument('-ds', '--tdownsamp', type=int, default=-1)
    args.add_argument('--mask', type=str, default=None, help='通道掩膜文件的路径')
    args = args.parse_args()
    return args


def handle_regular(data_path, retext):
    retexts = braceexpand(retext)
    file_list = []
    for expr in retexts:
        globi = glob(data_path + expr)
        file_list.extend(globi)
    file_list = np.sort(file_list)
    return file_list


def predict(model_session, data, prob=0.5):
    # ONNX Runtime 需要 numpy 数组作为输入
    inputs = np.expand_dims(data, axis=1).astype(np.float32, copy=False)
    
    # 执行推理
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    predict_res = model_session.run([output_name], {input_name: inputs})[0]
    
    # 后处理结果 (Softmax 不包含在导出的模型中)
    exp_scores = np.exp(predict_res)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    predict_res = softmax_probs[:, 1]
    blocks = np.where(predict_res >= prob)[0]
    return blocks


def model_load(base_model, device):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(base_model, options, 
                                    providers=['CPUExecutionProvider'])
    return model


def main(file_name, data, offset_base, file_info, model_session, prob,
                       ds_dds, ds_chunk, tdownsamp, plot_executor, save_path,
                       block_size, time_reso, mask_block_indices=None):
    """通用数据处理、预测和绘图提交例程。

    输入 Inputs:
    - file_name: 正在处理的文件路径
    - data: 该文件的原始数据块 (将在内部进行消色散)
    - offset_base: 绘图时添加的基础时间偏移量 (秒)
    - file_info: 元组 (时间分辨率, 频率分辨率, ..., 文件长度, 频率)
    - model_session: ONNX runtime 会话
    - prob: 候选块的概率阈值
    - ds_dds, ds_chunk, tdownsamp: 消色散/降采样参数
    - plot_executor: 提交绘图作业的执行器
    - save_path, block_size, time_reso: 用于绘图的其他全局变量

    返回检测到的块数量。
    """
    # 消色散 / 降采样
    new_data = dedisperse(data, ds_dds, ds_chunk, use_numba=True)
    n_time, n_freq = new_data.shape
    
    # 动态 1:1 切片
    # 我们希望块的持续时间 (以时间 bin 为单位) 等于 n_freq 以保持 1:1 的纵横比
    block_len = n_freq
    
    blocks_list = []
    offsets_list = []
    
    # 由于 Dataloader 现在处理补齐，如果 chunk_size 配置正确，n_time 理想情况下应该是 block_len 的倍数。
    # 但是，为了安全起见，我们仍然处理残余部分。
    indices = list(range(0, n_time, block_len))
    
    for idx in indices:
        start = idx
        end = idx + block_len
        
        # 如果最后一个块不完整，我们跳过或处理 (用户要求在 dataloader 中处理，所以这里还是要假设数据充足，或者只处理符合逻辑的部分)
        # 使用调整后的 chunk_size，end <= n_time 应该在大多数情况下成立。
        # 但如果总文件长度不能整除，dataloader 进行了循环补齐。
        # 所以我们可以相信有足够的数据，或者只取符合逻辑的部分。
        
        chunk_cut = None
        if end <= n_time:
            chunk_cut = new_data[start:end, :]
            offsets_list.append(start)
        else:
            # 要求 DataProc / ds_chunk 配置保证 n_time 可被 block_len 整除；
            # 这里不再兜底回退截取，以免引入重复块和时间语义混乱。
            break
        
        if chunk_cut is not None:
            # 使用 adaptive_avg_pool2d 进行调整大小以实现 "均值平滑"
            # 输入: (block_len, n_freq) -> (1, 1, block_len, n_freq)
            # 输出: (512, 512)
            t_data = torch.tensor(chunk_cut, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # adaptive_avg_pool2d 等效于降采样时的均值分箱 (binning)
            resized = F.adaptive_avg_pool2d(t_data, (512, 512))
            blocks_list.append(resized.squeeze().numpy())

    if len(blocks_list) == 0:
        return 0

    data_blocks = np.array(blocks_list)
    
    # 应用掩膜（若存在）(将掩膜块列设置为 0)
    if mask_block_indices is not None and len(mask_block_indices) > 0:
        data_blocks[:, :, mask_block_indices] = 0

    # 对每个块进行预处理
    for j in range(data_blocks.shape[0]):
        # 亮度封顶 (mu + 3*sigma)
        mu = np.mean(data_blocks[j, :, :])
        sigma = np.std(data_blocks[j, :, :])
        cap = mu + 3 * sigma
        data_blocks[j, :, :] = np.clip(data_blocks[j, :, :], None, cap)
        data_blocks[j, :, :] = preprocess_data(data_blocks[j, :, :])

    blocks = predict(model_session, data_blocks, prob)
    load = DataLoader(file_name)
    load.load_header()
    file_tstart = load.tstart
    
    n_detect = 0
    for block_idx in blocks:
        # Get true time offset from our offsets_list
        true_start_index = offsets_list[block_idx]
        offset_block = (true_start_index) * time_reso * tdownsamp + offset_base
        
        # submit plotting job
        plot_executor.submit(plot_burst, (data_blocks[block_idx], file_tstart), file_name,
                                offset_block, file_info, tdownsamp, save_path)
        n_detect += 1
        
    return n_detect


if __name__ == "__main__":
    args = get_args()
    DM = args.dm
    data_path = args.input
    save_path = args.output
    prob = args.prob
    ncpus = 5
    plot_executor = ProcessPoolExecutor(max_workers=ncpus)
    file_list = handle_regular(data_path, args.re)
    print(f"{len(file_list)} file(s) in list.")
    loader = DataLoader(file_list[0])
    file_info = loader.get_params()
    time_reso, freq_reso, _, file_len, freq = file_info
    if args.tdownsamp > 0:
        tdownsamp = args.tdownsamp
    else:
        al = int(np.log2(0.4e-3/time_reso))
        tdownsamp = 2**al
        
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    dds  = (4148808.0 * DM * (freq**-2 - freq.max()**-2) 
                                /1000 /time_reso).astype(np.int64)
    dds_size = int(dds.max())
    # 降采样前每次读取的原始样本数
    # 由时间采样点数决定
    nominal_chunk = 1536 * 512
    # 调整 chunk_size 为 (freq_reso * tdownsamp) 的倍数以确保整除
    unit = freq_reso * tdownsamp
    chunk_size = max(unit, int(round(nominal_chunk / unit) * unit))

    if file_len <= chunk_size:
        chunk_size = -1
        total_chunk = len(file_list) 
        # File-mode: ensure dedisperse output length is divisible by freq_reso (1:1 block_len).
        ds_chunk_raw = int(file_len // tdownsamp)
        ds_chunk = int((ds_chunk_raw // freq_reso) * freq_reso)
        if ds_chunk <= 0:
            # Extremely short files: fall back to one block.
            ds_chunk = int(freq_reso)
    else:
        print(f'Processing data by chunk size:{chunk_size//512}x512 (adjusted for freq={freq_reso}).')
        total_chunk = np.ceil((len(file_list) * file_len) / chunk_size).astype(int)
        ds_chunk = chunk_size // tdownsamp
    # 创建预加载数据队列
    preload_queue = mp.Queue(maxsize=min(4, total_chunk//2))
    
    global_mask_indices = None
    if args.mask:
        mask_chans = load_mask(args.mask)
        if mask_chans is not None:
             # Map raw channel indices to block column indices (512 columns)
             # Block column j corresponds to raw channels [j*factor, (j+1)*factor)
             # factor = freq_reso / 512
             factor = freq_reso / 512.0
             global_mask_indices = np.unique((mask_chans / factor).astype(int))
             # Ensure indices are within [0, 512)
             global_mask_indices = global_mask_indices[(global_mask_indices >= 0) & (global_mask_indices < 512)]
             print(f"Global Mask loaded: {len(mask_chans)} channels mapped to {len(global_mask_indices)} block columns.")

    preload_process = mp.Process(target=preload_worker, args=(
    file_list, chunk_size, dds_size, tdownsamp, freq_reso, ds_chunk, preload_queue))
    preload_process.start()
    data_source = preload_queue
    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
    base_model = './class_resnet18.onnx'
    model = model_load(base_model, device)
    chunk_idx = 0
    current_file_idx = -1
    current_mask_indices = None
    
    while True:
        item = data_source.get()
        if item is None:
            break
        file_idx, data_chunk = item
        data = data_chunk
        file_name = file_list[file_idx]
        basename = os.path.basename(file_name)
        
        # Determine mask
        if global_mask_indices is not None:
            mask_block_indices = global_mask_indices
        else:
            if file_idx != current_file_idx:
                current_file_idx = file_idx
                # Look for specific mask
                mask_path = file_name.replace('.fits', '.mask')
                if not os.path.exists(mask_path):
                     # Try alternative naming or location if needed
                     pass
                
                if os.path.exists(mask_path):
                    m_chans = load_mask(mask_path)
                    if m_chans is not None:
                        factor = freq_reso / 512.0
                        m_blks = np.unique((m_chans / factor).astype(int))
                        current_mask_indices = m_blks[(m_blks >= 0) & (m_blks < 512)]
                        print(f"Loaded mask for {basename}: {len(current_mask_indices)} masked blocks")
                    else:
                        current_mask_indices = None
                else:
                    current_mask_indices = None
            mask_block_indices = current_mask_indices

        if chunk_size > 0:
            offset = chunk_idx * chunk_size * time_reso
            progress_str = f"{chunk_idx+1}/{total_chunk}"
            chunk_str = f", chunk idx {chunk_idx}"
            chunk_idx += 1
        else:
            # Use global-time offset (seconds since first file start) for consistent timing across files.
            dl = DataLoader(file_name)
            _dt_i, _nchan_i, tstart_i, _file_len_i, _freq_i = dl.get_params()
            offset = (tstart_i - file_info[2]) * 86400.0
            progress_str = f"{file_idx+1}/{total_chunk}"
            chunk_str = ""
            file_idx += 1
        print(f"{progress_str}, file: {basename}")
        n_found = main(file_name, data, offset, file_info, model,
                                    prob, ds_dds, ds_chunk, tdownsamp,
                                    plot_executor, save_path, block_size, time_reso, mask_block_indices)
        print(f"Find {n_found} candidates in file {basename}{chunk_str}")
    preload_process.join()  # Wait for the preload process to finish
    plot_executor.shutdown(wait=True)
    end_time = datetime.now()
    print(f"Total processing time: {(end_time - strt_time).seconds} s")