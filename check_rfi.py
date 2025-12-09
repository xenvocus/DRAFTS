import os

# We set the cache directory to a local writable folder to prevent Numba caching errors.
if 'NUMBA_CACHE_DIR' not in os.environ:
    numba_cache_dir = os.path.join(os.getcwd(), 'numba_cache')
    try:
        os.makedirs(numba_cache_dir, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir
    except Exception:
        # If we can't create a cache dir, disable caching
        os.environ['NUMBA_DISABLE_CACHE'] = '1'

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from glob import glob
from braceexpand import braceexpand
from DataProc import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Quickly check RFI masking effect by plotting a random chunk.")
    parser.add_argument('-i', '--input', type=str, default=None, help='Input file pattern (support brace expansion)')
    parser.add_argument('-re', type=str, default='*.fits', help='Recursive file pattern (for compatibility)')
    parser.add_argument('-dm', '--dm', type=float, default=0, help='Dummy DM (ignored, for compatibility)')
    parser.add_argument('--mask', type=str, default=None, help='Path to channel mask file')
    parser.add_argument('-n', '--length', type=int, default=2048, help='Number of time samples to read')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    parser.add_argument('-o', '--output', type=str, default='check_rfi.png', help='Output image filename')
    return parser.parse_args()

def handle_regular(data_path):
    retexts = braceexpand(data_path)
    file_list = []
    for expr in retexts:
        globi = glob(expr)
        file_list.extend(globi)
    file_list = np.sort(file_list)
    return file_list

def main():
    args = get_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 1. Get file list
    input_pattern = args.input if args.input else args.re
    file_list = handle_regular(input_pattern)
    if len(file_list) == 0:
        print(f"No files found matching {input_pattern}")
        return

    # 2. Randomly select a file
    filename = random.choice(file_list)
    print(f"Selected file: {filename}")

    # 3. Initialize Loader to get file info
    loader = DataLoader(filename, mask_file=args.mask)
    header = loader.load_header()
    # file_len is total time samples
    file_len = loader.file_len
    nchans = loader.freq_reso
    
    print(f"File length: {file_len} samples")
    print(f"Channels: {nchans}")
    if args.mask:
        print(f"Applying mask from: {args.mask}")
        if loader.mask_chans is not None:
            print(f"Masked {len(loader.mask_chans)} channels.")

    # 4. Randomly select start position
    # Ensure we don't go out of bounds
    max_start = max(0, file_len - args.length)
    start = random.randint(0, max_start)
    print(f"Reading {args.length} samples from offset {start}")

    # 5. Load data (Mask is applied inside load())
    data = loader.load(start=start, length=args.length)
    # data shape: (Time, Pol, Chan). We usually take Pol 0 or sum/mean if multiple pols exist.
    # Current DataLoader implementation seems to keep (Time, Pol, Chan).
    # Let's inspect the shape
    print(f"Data shape: {data.shape}")
    
    # Flatten Pols if needed (e.g. Stokes I) or just take first index
    if data.ndim == 3:
        # (Time, Pol, Chan) -> (Time, Chan)
        # Using mean across Pol dimension or just taking 0
        data_2d = np.mean(data, axis=1)
    else:
        data_2d = data

    # 6. Plotting
    # Transpose to (Freq, Time) for standard waterfall plot convention
    plot_data = data_2d.T 
    
    # Calculate simple stats
    mean_prof = np.mean(plot_data, axis=0)
    bandpass = np.mean(plot_data, axis=1)

    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(3, 3, wspace=0.05, hspace=0.05)

    # Main Waterfall: (1:, :2) (Bottom Left block)
    ax_main = fig.add_subplot(gs[1:, :2])
    # Use simple normalization for visualization
    mean_val = np.mean(plot_data)
    std_val = np.std(plot_data)
    vmin = mean_val - 3 * std_val
    vmax = mean_val + 5 * std_val
    
    im = ax_main.imshow(plot_data, aspect='auto', origin='lower', cmap='viridis', 
                   vmin=vmin, vmax=vmax, extent=[start, start+args.length, 0, nchans])
    ax_main.set_xlabel('Time Sample Index')
    ax_main.set_ylabel('Frequency Channel')

    # Top Profile: (0, :2)
    ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
    ax_top.plot(np.arange(start, start+args.length), mean_prof, color='black', linewidth=0.8)
    ax_top.set_ylabel('Intensity')
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title(f'File: {os.path.basename(filename)}\nOffset: {start}, Mask: {args.mask is not None}')

    # Right Bandpass: (1:, 2)
    ax_right = fig.add_subplot(gs[1:, 2], sharey=ax_main)
    ax_right.plot(bandpass, np.arange(nchans), color='black', linewidth=0.8)
    ax_right.set_xlabel('Intensity')
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Save
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
