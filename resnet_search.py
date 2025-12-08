import os
import math
import torch
import warnings
import argparse
import numpy as np
import seaborn as sns
import onnxruntime as ort 
import multiprocessing as mp
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from braceexpand import braceexpand
from DataProc import DataLoader, preload_worker
from concurrent.futures import ProcessPoolExecutor
from DataProc.utils import preprocess_data, dedisperse, plot_burst, data_padding

try:
    import psutil
except ImportError:
    psutil = None

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
    # ONNX Runtime expects numpy array as input
    inputs = np.expand_dims(data, axis=1).astype(np.float32, copy=False)
    
    # Run inference
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    predict_res = model_session.run([output_name], {input_name: inputs})[0]
    
    # Post-process the result (softmax is not part of the exported model)
    logits = predict_res - np.max(predict_res, axis=1, keepdims=True)
    exp_scores = np.exp(logits, dtype=np.float32)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    predict_res = softmax_probs[:, 1]
    blocks = np.where(predict_res >= prob)[0]
    return blocks


def model_load(base_model, device):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    cpu_threads = max(1, (os.cpu_count() or 1) - 1)
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = min(cpu_threads, 4)

    providers = []
    provider_options = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # 默认使用首个 GPU，可根据需要扩展为多 GPU
        providers.append('CUDAExecutionProvider')
        provider_options.append({'device_id': 0})
    providers.append('CPUExecutionProvider')

    model = ort.InferenceSession(base_model, options,
                                 providers=providers,
                                 provider_options=provider_options if provider_options else None)
    return model


def preprocess_blocks(data_blocks, exp_cut=5):
    """Vectorized preprocessing over all blocks to reduce Python-loop overhead."""
    # data_blocks: (nb, t, f)
    data_blocks = data_blocks.astype(np.float32, copy=False)
    data_blocks += 1.0
    data_blocks /= np.mean(data_blocks, axis=1, keepdims=True)
    vmins = np.nanpercentile(data_blocks, exp_cut, axis=(1, 2))
    vmaxs = np.nanpercentile(data_blocks, 100 - exp_cut, axis=(1, 2))
    vmins = vmins.reshape(-1, 1, 1)
    vmaxs = vmaxs.reshape(-1, 1, 1)
    np.clip(data_blocks, vmins, vmaxs, out=data_blocks)
    min_vals = data_blocks.min(axis=(1, 2), keepdims=True)
    max_vals = data_blocks.max(axis=(1, 2), keepdims=True)
    data_blocks -= min_vals
    data_blocks /= (max_vals - min_vals + 1e-6)
    return data_blocks


def main(file_name, data, offset_base, file_info, model_session, prob,
                       ds_dds, ds_chunk, tdownsamp, plot_executor, save_path,
                       block_size, time_reso):
    """Common data processing, prediction and plot submission routine.

    Inputs:
    - file_name: path to the file being processed
    - data: raw data chunk for this file (will be dedispersed inside)
    - offset_base: base time offset (seconds) to add for plotting
    - file_info: tuple (time_reso, freq_reso, ..., file_len, freq)
    - model_session: ONNX runtime session
    - prob: probability threshold for candidate blocks
    - ds_dds, ds_chunk, tdownsamp: dedispersion/downsample parameters
    - plot_executor: executor to submit plotting jobs
    - save_path, block_size, time_reso: additional globals used for plotting

    Returns number of detected blocks.
    """
    # Dedisperse / downsample
    new_data = dedisperse(data, ds_dds, ds_chunk, use_numba=True)
    data_padded = data_padding(new_data)
    t, f = data_padded.shape
    # reshape into blocks of 512x512 (time-block x 512 x freq-blocks)
    data_blocks = np.mean(data_padded.reshape(t//512, 512, 512, f//512), axis=3)
    # preprocess per block (vectorized)
    data_blocks = preprocess_blocks(data_blocks)

    blocks = predict(model_session, data_blocks, prob)
    load = DataLoader(file_name)
    load.load_header()
    file_tstart = load.tstart
    for block in blocks:
        offset_block = (block * block_size) * time_reso * tdownsamp + offset_base
        # submit plotting job; keep call signature unchanged
        plot_executor.submit(plot_burst, (data_blocks[block], file_tstart), file_name,
                                offset_block, file_info, tdownsamp, save_path)
    return len(blocks)


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
    # Raw samples to read in each step before downsampling
    # 基准 chunk，大内存时可调大以减少进程切换
    base_chunk = 1536 * 512
    chunk_size = base_chunk

    if psutil is not None:
        vm = psutil.virtual_memory()
        avail_bytes = vm.available
        # 每个样本的字节估计：2 极化 * freq_reso * float32
        bytes_per_sample = 2 * freq_reso * 4
        target_bytes = avail_bytes * 0.10  # 10% 可用内存作为上限
        max_samples_mem = int(target_bytes // bytes_per_sample)
        # 对齐到 512，限制在基准的 4 倍以内以兼顾吞吐与内存
        max_samples_mem = max(512, (max_samples_mem // 512) * 512)
        chunk_size = min(max(base_chunk, max_samples_mem), base_chunk * 4)
    
    # 预估总采样数（不加载数据，只读头），用于更准确的进度估计
    total_samples = 0
    for f in file_list:
        _dl = DataLoader(f)
        _dl.load_header()
        total_samples += _dl.file_len

    if file_len <= chunk_size:
        chunk_size = -1
        total_chunk = len(file_list) 
        ds_chunk = file_len // tdownsamp
    else:
        print(f'Processing data by chunk size:{chunk_size//512}x512.')
        # 包含去色散重叠的上界估计，避免末尾补零导致的低估
        total_chunk = max(1, math.ceil((total_samples + dds_size) / chunk_size))
        ds_chunk = chunk_size // tdownsamp
    # Create a queue for preloading data
    preload_queue = mp.Queue(maxsize=min(4, total_chunk//2))
    preload_process = mp.Process(target=preload_worker, args=(
    file_list, chunk_size, dds_size, tdownsamp, freq_reso, ds_chunk, preload_queue))
    preload_process.start()
    data_source = preload_queue
    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
    base_model = './class_resnet18.onnx'
    model = model_load(base_model, device)

    # 预热 numba 去色散内核，避免首块编译开销
    ds_dds_max = int(ds_dds.max()) if ds_dds.size > 0 else 0
    warm_len = ds_chunk + ds_dds_max
    warm_arr = np.zeros((warm_len, freq_reso), dtype=np.float32)
    dedisperse(warm_arr, ds_dds, ds_chunk, use_numba=True)
    chunk_idx = 0
    while True:
        item = data_source.get()
        if item is None:
            break
        file_idx, data_chunk = item
        data = data_chunk
        file_name = file_list[file_idx]
        basename = os.path.basename(file_name)
        if chunk_size > 0:
            offset = chunk_idx * chunk_size * time_reso
            progress_str = f"{chunk_idx+1}/{total_chunk}"
            chunk_str = f", chunk idx {chunk_idx}"
            chunk_idx += 1
        else:
            offset = 0
            progress_str = f"{file_idx+1}/{total_chunk}"
            chunk_str = ""
            file_idx += 1
        print(f"{progress_str}, file: {basename}")
        n_found = main(file_name, data, offset, file_info, model,
                                    prob, ds_dds, ds_chunk, tdownsamp,
                                    plot_executor, save_path, block_size, time_reso)
        print(f"Find {n_found} candidates in file {basename}{chunk_str}")
    preload_process.join()  # Wait for the preload process to finish
    plot_executor.shutdown(wait=True)
    end_time = datetime.now()
    print(f"Total processing time: {(end_time - strt_time).seconds} s")