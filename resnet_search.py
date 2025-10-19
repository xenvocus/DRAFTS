import os
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
    args.add_argument('-p', '--prob', type=float, default=0.6)
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
    # preprocess per block
    for j in range(data_blocks.shape[0]):
        data_blocks[j, :, :] = preprocess_data(data_blocks[j, :, :])

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
    # Determined by nsamps of time 
    chunk_size = 1536 * 512
    
    if file_len <= chunk_size:
        chunk_size = -1
        total_chunk = len(file_list) 
        ds_chunk = file_len // tdownsamp
    else:
        print(f'Processing data by chunk size:{chunk_size//512}x512.')
        total_chunk = np.ceil((len(file_list) * file_len) / chunk_size).astype(int)
        ds_chunk = chunk_size // tdownsamp
    # Create a queue for preloading data
    preload_queue = mp.Queue(maxsize=min(4, total_chunk//2))
    preload_process = mp.Process(target=preload_worker, args=(
    file_list, chunk_size, dds_size, tdownsamp, freq_reso, preload_queue))
    preload_process.start()
    data_source = preload_queue
    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
    base_model = './class_resnet18.onnx'
    model = model_load(base_model, device)
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