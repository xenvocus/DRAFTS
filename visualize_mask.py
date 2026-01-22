import os
import argparse
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
sns.set_color_codes()
# Setup environment like resnet_search.py
if 'NUMBA_CACHE_DIR' not in os.environ:
    numba_cache_dir = os.path.join(os.getcwd(), 'numba_cache')
    os.makedirs(numba_cache_dir, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir

from DataProc import DataLoader
from DataProc.utils import preprocess_data, dedisperse, plot_burst, load_mask

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Visualize Mask application (Consistent with resnet_search.py)")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input FITS/FIL file')
    parser.add_argument('-dm', '--dm', type=float, default=893, help='Dispersion Measure (should match search)')
    parser.add_argument('--mask', type=str, default=None, help='Path to mask file')
    parser.add_argument('-o', '--output', type=str, default='./mask_viz', help='Output directory')
    parser.add_argument('-n', '--num_plots', type=int, default=3, help='Number of random chunks to plot')
    parser.add_argument('-ds', '--tdownsamp', type=int, default=-1, help='Time downsampling factor')
    return parser.parse_args()

def process_chunk(data, file_info, tdownsamp, mask_block_idc, ds_dds, ds_chunk):
    """
    Process a chunk of data exactly as resnet_search.py does.
    Returns processed blocks ready for plotting.
    """
    # 0. Pre-processing: Downsample
    # Data is raw (Time, Pol, Chan) or (Time, Chan)
    # Ensure dimensions
    if data.ndim == 3:
        n_time, n_pol, n_chan = data.shape
        # Downsample time and average pols
        ds_len = n_time // tdownsamp
        if ds_len == 0: return [], []
        
        reshaped = data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, n_pol, n_chan)
        data_ds = np.mean(reshaped, axis=(1, 2)).astype(np.float32)
    elif data.ndim == 2:
        # (Time, Chan)
        n_time, n_chan = data.shape
        ds_len = n_time // tdownsamp
        if ds_len == 0: return [], []
        reshaped = data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, n_chan)
        data_ds = np.mean(reshaped, axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # 1. Dedisperse
    # data_ds is now (Time_ds, Chan)
    new_data = dedisperse(data_ds, ds_dds, ds_chunk, use_numba=True)
    n_time, n_freq = new_data.shape
    
    # 2. Block slicing (1:1 aspect ratio)
    block_len = n_freq
    blocks_list = []
    offsets_list = []
    
    # Just take the first valid block for visualization to keep it simple, 
    # or iterate like the original code. 
    # Since we control the read size, let's try to get a few blocks.
    
    idc = list(range(0, n_time, block_len))
    
    for idx in idc:
        start = idx
        end = idx + block_len
        
        if end <= n_time:
            chunk_cut = new_data[start:end, :]
            
            # 3. Resize / Downsample to 512x512
            t_data = torch.tensor(chunk_cut, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized = F.adaptive_avg_pool2d(t_data, (512, 512))
            block_numpy = resized.squeeze().numpy()
            
            blocks_list.append(block_numpy)
            offsets_list.append(start)
            
    if not blocks_list:
        return [], []

    data_blocks = np.array(blocks_list)
    
    # 4. Apply Mask
    if mask_block_idc is not None and len(mask_block_idc) > 0:
        # EXACT LOGIC FROM resnet_search.py provided in context
        data_blocks[:, :, mask_block_idc] = np.mean(data_blocks)
        
    # 5. Preprocess (Normalize)
    # Note: resnet_search.py does preprocessing AFTER masking.
    for j in range(data_blocks.shape[0]):
        data_blocks[j, :, :] = preprocess_data(data_blocks[j, :, :])
        
    return data_blocks, offsets_list

def main():
    args = get_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    loader = DataLoader(args.input)
    file_info = loader.get_params()
    time_reso, freq_reso, tstart, file_len, freq = file_info
    
    print(f"File: {args.input}")
    print(f"Length: {file_len} samples, Freq chans: {freq_reso}")
    
    # Determine tdownsamp
    if args.tdownsamp > 0:
        tdownsamp = args.tdownsamp
    else:
        al = int(np.log2(0.4e-3/time_reso))
        tdownsamp = 2**al
    print(f"Time Downsample Factor: {tdownsamp}")

    # Prepare dedispersion params
    dds  = (4148808.0 * args.dm * (freq**-2 - freq.max()**-2) 
                                /1000 /time_reso).astype(np.int64)
    dds_size = int(dds.max())
    
    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
    
    # Chunk size logic (simplified for visualization)
    # We want to read enough for at least one block + dedispersion padding
    # Block duration in raw samples = freq_reso * tdownsamp
    one_block_raw_time = freq_reso * tdownsamp
    
    # Read enough for ~2 blocks to be safe and have variety
    read_len_time = one_block_raw_time * 2
    
    # Padding required for dedispersion
    # In resnet_search.py logic, the 'data' passed to main includes the padding
    # needed for dedispersion shift. The dedisperse function shifts data.
    # The max shift is dds_size.
    read_len = int(read_len_time + dds_size)
    
    # ds_chunk needed for dedisperse function
    # In resnet_search, ds_chunk is the effective length after downsampling
    # It seems used for optimizing loop or index, but dedisperse function signature is:
    # dedisperse(data, ds_dds, ds_chunk, use_numba=True)
    # Let's check dedisperse implementation or usage. 
    # In resnet_search main:
    # new_data = dedisperse(data, ds_dds, ds_chunk, use_numba=True)
    # ds_chunk logic in resnet_search is complicated.
    # However, if we read a single standalone chunk, we can set ds_chunk to reflect the valid output size.
    # Let's set ds_chunk to output size
    ds_chunk = int(read_len_time // tdownsamp)

    # Load Mask
    mask_block_idc = None
    if args.mask:
        mask_chans = load_mask(args.mask)
        if mask_chans is not None:
             factor = freq_reso / 512.0
             global_mask_idc = np.unique((mask_chans / factor).astype(int))
             mask_block_idc = global_mask_idc[(global_mask_idc >= 0) & (global_mask_idc < 512)]
             print(f"Mask loaded: {len(mask_block_idc)} blocked columns (512-width)")

    # Plot Executor (dummy, just to run function)
    # But plot_burst is a standalone function, we can just call it directly.
    
    # Randomly pick N spots
    rng = np.random.default_rng()
    max_start = max(0, file_len - read_len)
    
    starts = rng.choice(max_start, size=min(args.num_plots, max_start//read_len + 1), replace=False)
    starts.sort()
    
    print(f"Generating {len(starts)} plots...")
    
    for i, start_idx in enumerate(starts):
        print(f"Processing chunk {i+1} at offset {start_idx}...")
        
        # Load raw data
        # Note: DataLoader.load takes start, length
        # resnet_search uses a generator/queue, but essentially calls loader.load underneath via fitsio
        raw_data = loader.load(start=start_idx, length=read_len)
        
        # Process
        # We pass ds_chunk same as calculated above.
        # Note: raw_data might be slightly smaller if at EOF, but we limited max_start.
        
        data_blocks, offsets = process_chunk(raw_data, file_info, tdownsamp, mask_block_idc, ds_dds, ds_chunk)
        
        if len(data_blocks) == 0:
            print("No valid blocks generated from this chunk.")
            continue
            
        # Plot the first valid block from this chunk
        blk = data_blocks[0]
        # Calculate true time offset
        # offset_base is the time offset in seconds for the start of the file or chunk
        # Here we need absolute time offset from file start.
        # start_idx is in raw samples.
        # time_reso is raw sample time.
        
        # The offset passed to plot_burst should be:
        # (start_idx / file_len_samples * total_time)? No.
        # It is: start_idx * time_reso
        # BUT, we must account for the shift inside process_chunk.
        # offsets[0] is the index in the dedispersed/downsampled array.
        
        # In resnet_search:
        # offset = chunk_idx * chunk_size * time_reso (Base offset of the chunk)
        # true_start_index = offsets_list[block_idx] (Index within the chunk)
        # offset_block = (true_start_index) * time_reso * tdownsamp + offset_base
        
        base_time_offset = start_idx * time_reso
        internal_offset_idx = offsets[0]
        
        final_offset = (internal_offset_idx * time_reso * tdownsamp) + base_time_offset
        
        # Call plot_burst
        # Signature: plot_burst((data, tstart), file_name, offset, file_info, tdownsamp, save_path)
        # We use a dummy executor or just call it.
        try:
            plot_burst((blk, tstart), args.input, final_offset, file_info, tdownsamp, args.output)
            print(f"Plot saved for offset {final_offset:.4f}s")
        except Exception as e:
            print(f"Error plotting: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
