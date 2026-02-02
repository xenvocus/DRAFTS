import os
import re # Added for filename sanitization

# 设置缓存目录到本地可写文件夹
if 'NUMBA_CACHE_DIR' not in os.environ:
    numba_cache_dir = os.path.join(os.getcwd(), 'numba_cache')
    os.makedirs(numba_cache_dir, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir

import torch
import onnx
import onnx.numpy_helper
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


def model_load_advanced(model_path):
    """
    Load ONNX model, extract FC weights, and modify model to output feature map.
    Returns: (session, fc_weights, feature_layer_name)
    """
    model = onnx.load(model_path)
    
    # Target nodes based on inspection
    feature_node_name = '/base_model/layer4/layer4.1/relu_1/Relu_output_0'
    fc_weight_name = 'base_model.fc.weight'
    
    # Extract FC weights
    fc_weights = None
    for init in model.graph.initializer:
        if init.name == fc_weight_name:
            fc_weights = onnx.numpy_helper.to_array(init)
            break
            
    if fc_weights is None:
        print(f"Warning: Could not find FC weights {fc_weight_name} in ONNX model.")
        # Fallback or error handling
    
    # Add intermediate output
    # Check if it's already an output
    if not any(out.name == feature_node_name for out in model.graph.output):
        # Create ValueInfoProto for the output (we can infer type/shape or leave partially undefined)
        # Usually minimal info is enough for runtime to fill it
        intermediate_layer_value_info = onnx.helper.make_tensor_value_info(
            feature_node_name,
            onnx.TensorProto.FLOAT,
            ['batch', 'channel', 'height', 'width'] # Symbolic dims
        )
        model.graph.output.append(intermediate_layer_value_info)
        
    # Create session from bytes
    model_bytes = model.SerializeToString()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_bytes, options, providers=['CPUExecutionProvider'])
    
    return session, fc_weights, feature_node_name

def get_cam_bbox(session, data, fc_weights, feature_output_name, threshold_ratio=0.5):
    try:
        # Prepare input
        input_name = session.get_inputs()[0].name
        # Input expects (Batch, Channel, H, W) -> (1, 1, 512, 512)
        inputs = np.expand_dims(data, axis=0).astype(np.float32) # (1, 512, 512)
        inputs = np.expand_dims(inputs, axis=0) # (1, 1, 512, 512)
        
        # Run session to get feature map
        # Request only the feature map output to save time? Or both?
        # Note: If we just run, we get all outputs.
        outputs = session.run([feature_output_name], {input_name: inputs})
        feature_map = outputs[0] # (1, 512, H, W) -> usually (1, 512, 16, 16) for ResNet18 input 512
        
        if fc_weights is None:
            return None

        # CAM calculation
        # feature_map: (1, 512, 16, 16)
        # fc_weights: (2, 512) -> We want class 1 (signal)
        weight = fc_weights[1] # (512,)
        
        # (1, 512, H, W) * (512,) -> (1, H, W)
        # np.einsum is convenient: 'bchw,c->bhw'
        cam = np.einsum('bchw,c->bhw', feature_map, weight)
        
        # Normalize and Resize
        # Use torch for interpolation as it's already imported and easy
        cam_tensor = torch.from_numpy(cam).unsqueeze(1) # (1, 1, H, W)
        cam_resized = F.interpolate(cam_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        cam_np = cam_resized.squeeze().numpy() # (512, 512)
        
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-8)
        
        mask = cam_np > threshold_ratio
        
        if not np.any(mask):
            return None
            
        t_indices = np.where(np.any(mask, axis=1))[0] # Time is dim 0 (height in array)
        f_indices = np.where(np.any(mask, axis=0))[0] # Freq is dim 1 (width in array)
        
        # In plot_burst: imshow(data.T)
        # data is (Time, Freq) -> (512, 512)
        # imshow(data.T) means X-axis is Time (dim 0 of original), Y-axis is Freq (dim 1 of original)
        # But wait, imshow(data.T) puts dim 1 (Freq) on Y, dim 0 (Time) on X.
        
        # bbox format expected by plot_burst in utils.py:
        # x_min, x_max, y_min, y_max
        # In plot_burst:
        # rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min)
        # plt.imshow(data.T, ...) 
        # data.T shape is (Freq, Time). 
        # imshow uses (row, col) as (y, x).
        # data.T[y, x] corresponds to data[x, y].
        # So X-coord in plot is Time index (0..511). Y-coord in plot is Freq index (0..511).
        
        # t_indices are indices in dimension 0 of data (Time).
        # f_indices are indices in dimension 1 of data (Freq).
        
        if len(t_indices) == 0 or len(f_indices) == 0:
            return None
            
        x_min, x_max = int(t_indices[0]), int(t_indices[-1])
        y_min, y_max = int(f_indices[0]), int(f_indices[-1])
        
        return (x_min, x_max, y_min, y_max)
        
    except Exception as e:
        print(f"Error computing CAM: {e}")
        return None


def clean_block(data, threshold=0.05, max_iter=None, return_mask=False):
    """
    Apply FFT-based RFI masking to frequency channels.
    Identify channels with high modulation in FFT domain and mask them.
    
    threshold: Percentage (0-1) of channels to cut. Default 0.05 (5%).
               The algorithm masks channels whose max FFT magnitude is in the top `threshold` %.
    """
    data = data.copy()
    n_time, n_freq = data.shape
    
    # 1. Compute FFT along time axis (axis 0)
    # Skip DC component (index 0)
    fft_data = np.fft.fft(data, axis=0)[1:] 
    
    # 2. Get Magnitude
    mag = np.abs(fft_data)
    
    # 3. Find max magnitude for each channel across frequencies (time-freqs)
    # axis=0 here reduces the time dim (which is now freq domain of time series)
    max_freq_mag = np.max(mag, axis=0) # shape (n_freq,)
    
    # 4. Determine Threshold
    # We zap the top `threshold` percent of channels
    fft_thres_val = (1 - threshold) * 100
    cutoff = np.nanpercentile(max_freq_mag, fft_thres_val)
    
    # 5. Create Mask
    # Mask channels with magnitude > cutoff
    new_mask = max_freq_mag > cutoff
    
    # 6. Apply Mask
    # Replace masked channels with global median
    if np.any(new_mask):
        global_med = np.median(data)
        data[:, new_mask] = global_med
        
    mask_indices = np.where(new_mask)[0]
    
    if return_mask:
        return data, mask_indices
    return data


def main(file_name, data, offset_base, file_info, model_session, prob,
                       ds_dds, ds_chunk, tdownsamp, plot_executor, save_path,
                       time_reso, mask_block_idc=None, enable_dynamic_mask=False, 
                       fc_weights=None, feature_output_name=None):
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
    - save_path, time_reso: 用于绘图的其他全局变量

    返回检测到的块数量。
    """
    # 消色散 / 降采样
    new_data = dedisperse(data, ds_dds, ds_chunk, use_numba=True)
    n_time, n_freq = new_data.shape
    
    # 动态 1:1 切片 -> 修改：固定时间窗口 512
    # 我们希望块的持续时间 (以时间 bin 为单位) 等于 512
    block_len = 512
    
    blocks_list = []
    offsets_list = []
    
    # 由于 Dataloader 现在处理补齐，如果 chunk_size 配置正确，n_time 理想情况下应该是 block_len 的倍数。
    # 但是，为了安全起见，我们仍然处理残余部分。
    idc = list(range(0, n_time, block_len))
    
    for idx in idc:
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
            
            # 始终使用自适应平均池化进行降采样 (Binning)
            # 即使频率通道不是 512 的整数倍，也能通过平均保持通量守恒
            resized = F.adaptive_avg_pool2d(t_data, (512, 512))
            blocks_list.append(resized.squeeze().numpy())

    if len(blocks_list) == 0:
        return 0

    data_blocks = np.array(blocks_list)
    
    # 应用掩膜（若存在）
    if mask_block_idc is not None and len(mask_block_idc) > 0:
        data_blocks[:, :, mask_block_idc] = np.median(data_blocks)

    # 抽样检查 Mask 效果的标志
    check_plotted = False

    # 对每个块进行预处理
    for j in range(data_blocks.shape[0]):
        if enable_dynamic_mask:
            data_blocks[j, :, :], mask_idc = clean_block(data_blocks[j, :, :], return_mask=True)
            # 随机抽一张保存展示效果 (每个文件最多一张，概率 10% 以防错过短文件)
            if not check_plotted and np.random.rand() < 0.1:
                try:
                    plt.figure(figsize=(8, 8))
                    # Transpose to show Time on X, Freq on Y (Standard Waterfall)
                    plt.imshow(data_blocks[j, :, :].T, aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Dynamic Mask Check\nFile: {os.path.basename(file_name)}\nBlock: {j}")
                    plt.colorbar()
                    
                    # 用红线明确标明 mask 掉的 channel (Y 轴)
                    if mask_idc is not None:
                        for midx in mask_idc:
                             plt.axhline(y=midx, color='red', linewidth=0.8, alpha=0.7)

                    raw_name = f"mask_check_{os.path.basename(file_name).replace('.fits', '')}_blk{j}"
                    safe_name = re.sub(r'[<>:"/\\|?*]', '-', raw_name)
                    check_path = os.path.join(save_path, f"{safe_name}.jpg")
                    
                    plt.savefig(check_path)
                    plt.close()
                    print(f"Saved mask check image: {check_path}")
                    check_plotted = True
                except Exception as e:
                    print(f"Failed to plot mask check: {e}")

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
        
        bbox = None
        if feature_output_name is not None and fc_weights is not None:
             bbox = get_cam_bbox(model_session, data_blocks[block_idx], fc_weights, feature_output_name)

        # submit plotting job
        # 修正：为了画图时正确计算时间持续，传入的 "freq_reso" (这里被解释为 block_len) 必须是 512
        plot_file_info = list(file_info)
        plot_file_info[1] = 512
        plot_file_info = tuple(plot_file_info)
        
        plot_executor.submit(plot_burst, (data_blocks[block_idx], file_tstart), file_name,
                                offset_block, plot_file_info, tdownsamp, save_path, bbox)
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
    # 调整 chunk_size 为 (512 * tdownsamp) 的倍数以确保整除
    unit = 512 * tdownsamp
    chunk_size = max(unit, int(round(nominal_chunk / unit) * unit))

    if file_len <= chunk_size:
        chunk_size = -1
        total_chunk = len(file_list) 
        # File-mode: ensure dedisperse output length is divisible by 512 (fixed block_len).
        ds_chunk_raw = int(file_len // tdownsamp)
        ds_chunk = int((ds_chunk_raw // 512) * 512)
        if ds_chunk <= 0:
            # Extremely short files: fall back to one block.
            ds_chunk = 512
    else:
        print(f'Processing data by chunk size:{chunk_size//512}x512 (adjusted for fixed block size).')
        total_chunk = np.ceil((len(file_list) * file_len) / chunk_size).astype(int)
        ds_chunk = chunk_size // tdownsamp
    # 创建预加载数据队列
    preload_queue = mp.Queue(maxsize=min(4, total_chunk//2))
    
    global_mask_idc = None
    use_dynamic_mask = False
    
    if args.mask:
        if args.mask.lower() == 'auto':
            use_dynamic_mask = True
            print("Dynamic Block-level RFI Masking ENABLED.")
        else:
            mask_chans = load_mask(args.mask)
            if mask_chans is not None:
                # Map raw channel idc to block column idc (512 columns)
                # Block column j corresponds to raw channels [j*factor, (j+1)*factor)
                # factor = freq_reso / 512
                factor = freq_reso / 512.0
                global_mask_idc = np.unique((mask_chans / factor).astype(int))
                # Ensure idc are within [0, 512)
                global_mask_idc = global_mask_idc[(global_mask_idc >= 0) & (global_mask_idc < 512)]
                print(f"Global Mask loaded: {len(mask_chans)} channels mapped to {len(global_mask_idc)} block columns.")

    preload_process = mp.Process(target=preload_worker, args=(
    file_list, chunk_size, dds_size, tdownsamp, freq_reso, ds_chunk, preload_queue))
    preload_process.start()
    data_source = preload_queue
    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
    base_model = './class_resnet18.onnx'
    # Use advanced model loading to enable CAM
    model, fc_weights, feature_layer_name = model_load_advanced(base_model)
    print(f"Model loaded with CAM support. Feature layer: {feature_layer_name}")
    
    chunk_idx = 0
    current_file_idx = -1
    current_mask_idc = None
    
    while True:
        item = data_source.get()
        if item is None:
            break
        file_idx, data_chunk = item
        data = data_chunk
        file_name = file_list[file_idx]
        basename = os.path.basename(file_name)
        
        # Determine mask
        if global_mask_idc is not None:
            mask_block_idc = global_mask_idc
        else:
            if file_idx != current_file_idx:
                current_file_idx = file_idx
                # Look for specific mask
                if not use_dynamic_mask:
                    mask_path = file_name.replace('.fits', '.mask')
                    if not os.path.exists(mask_path):
                        # Try alternative naming or location if needed
                        pass
                    
                    if os.path.exists(mask_path):
                        m_chans = load_mask(mask_path)
                        if m_chans is not None:
                            factor = freq_reso / 512.0
                            m_blks = np.unique((m_chans / factor).astype(int))
                            current_mask_idc = m_blks[(m_blks >= 0) & (m_blks < 512)]
                            print(f"Loaded mask for {basename}: {len(current_mask_idc)} masked blocks")
                        else:
                            current_mask_idc = None
                    else:
                        current_mask_idc = None
            mask_block_idc = current_mask_idc

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
                                    plot_executor, save_path, time_reso, 
                                    mask_block_idc, enable_dynamic_mask=use_dynamic_mask,
                                    fc_weights=fc_weights, feature_output_name=feature_layer_name)
        print(f"Find {n_found} candidates in file {basename}{chunk_str}")
    preload_process.join()  # Wait for the preload process to finish
    plot_executor.shutdown(wait=True)
    end_time = datetime.now()
    print(f"Total processing time: {(end_time - strt_time).seconds} s")