import os
import fitsio
import argparse
import numpy as np
from datetime import datetime
import torch, torchvision
# from astropy.io import fits
from numba import njit, prange
from glob import glob
import seaborn as sns
from rich.progress import track
import matplotlib.pyplot as plt
from matplotlib import gridspec
from braceexpand import braceexpand
from sigpyproc.readers import FilReader
from BinaryClass.binary_model import SPPResNet, BinaryNet
import warnings
strt_time = datetime.now()
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_color_codes()
block_size = 512
tdownsamp = 4
base_model = 'resnet18'
model_path = './class_resnet18.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 1. auto adjust chunk_size according to memory size or file size
#       2. adjust tdownsamp according to time_reso, and allow user input


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-dm', '--dm', type=int, default=893)
    args.add_argument('-i', '--input', type=str, default='./')
    args.add_argument('-o', '--output', type=str, default='./')
    args.add_argument('-re', type=str, default='*.fits')
    args.add_argument('-p', '--prob', type=float, default=0.6)
    args = args.parse_args()
    return args


class DataLoader:
    def __init__(self, filename, telescope='Fake', backend='Fake'):
        self.filename = filename
        self.telescope = telescope
        self.backend = backend


    def load_fil_file(self, start, length):
        self.load_fil_header()
        fil = FilReader(self.filename)
        self.data = fil.read_block(start, length).astype(np.float32).T
        # assume no other pols
        self.data = self.data.reshape(-1, 1, fil.header.nchans) # [:, :2, :]
        return self.data


    def load_fits_file(self, start, length):
        self.load_fits_header()
        # transpose start time sample to (nsub, nsamp)
        nsblk = self.header['NSBLK']
        start_nsub = int(start/nsblk) 
        start_nsamp = start - start_nsub * nsblk
        end_nsub = np.ceil((start + length) / nsblk).astype(int) 
        end_nsamp = start_nsamp + length
        sub_idcs = np.arange(start_nsub, end_nsub)
        data, h = fitsio.read(self.filename, rows=sub_idcs, 
                                columns=['DATA'], ext=1, header=True)
        data = data['DATA'].astype(np.float32)
        data = data.reshape(-1, h['NPOL'], h['NCHAN'])[start_nsamp:end_nsamp, :2, :]
        data = np.mean(data, axis=1, keepdims=True)
        return data


    def load(self, start=0, length=None):
        ext = os.path.splitext(self.filename)[1]
        if ext == '.fil':
            self.data = self.load_fil_file(start, length)
        elif ext == '.fits':
            self.data = self.load_fits_file(start, length)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        self._reverse()
        return self.data


    def load_header(self):
        ext = os.path.splitext(self.filename)[1]
        if ext == '.fil':
            self.load_fil_header()
        elif ext == '.fits':
            self.load_fits_header()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        self._reverse()
        return self.header


    def _reverse(self):
        if self._freq_revflag:
            self.freq = self.freq[::-1]
            self.fch1 = self.freq[0]
            self._freq_revflag = False
        if hasattr(self, 'data') and hasattr(self, '_data_revflag'):
            if self._data_revflag:
                self.data = self.data[:, :, ::-1]
                self._data_revflag = False


    def load_fil_header(self):
        fil = FilReader(self.filename)
        self.header = fil.header
        self.time_reso = fil.header.tsamp
        self.freq_reso = int(fil.header.nchans)
        self.file_len = fil.header.nsamples
        self.fch1 = fil.header.fch1
        self.foff = fil.header.foff
        self.tstart = fil.header.tstart
        self.freq = self.fch1 + np.arange(self.freq_reso) * self.foff
        self._data_revflag = False if self.foff > 0 else True
        self._freq_revflag = False if self.foff > 0 else True
        self._reverse()
        del fil
        return self.header


    def load_fits_header(self):
        h0 = fitsio.read_header(self.filename)
        h = fitsio.read_header(self.filename, ext=1)
        self.header = h
        self.time_reso = h['TBIN']
        self.freq_reso = int(h['NCHAN'])
        self.file_len = h['NAXIS2'] * h['NSBLK']
        self.tstart = h0['STT_IMJD'] + h0['STT_SMJD'] / 86400 + h0['STT_OFFS'] / 86400 #MJD
        self.fch1 = h0['OBSFREQ'] - 0.5 * h0['OBSBW']
        self.foff = h['CHAN_BW']
        self.freq = self.fch1 + np.arange(self.freq_reso) * self.foff
        self._data_revflag = False if self.foff > 0 else True
        self._freq_revflag = False if self.foff > 0 else True
        self._reverse()
        return self.header


    def get_params(self):
        if hasattr(self, 'header'):
            return (self.time_reso, self.freq_reso, self.tstart, self.file_len, self.freq)
        else:
            self.load_header()
            return (self.time_reso, self.freq_reso, self.tstart, self.file_len, self.freq)


def handle_regular(data_path, retext):
    retexts = braceexpand(retext)
    file_list = []
    for expr in retexts:
        globi = glob(data_path + expr)
        file_list.extend(globi)
    file_list = np.sort(file_list)
    return file_list


@njit(parallel=True, fastmath=True)
def dedisperse(data, shifts, ds_chunk):
    shifts = np.asarray(shifts, dtype=np.int64)
    n_chan = data.shape[1]
    out = np.empty((ds_chunk, n_chan), dtype=np.float32)
    for j in prange(n_chan):
        s = shifts[j]
        for i in range(ds_chunk):
            out[i, j] = data[s + i, j]
    return out


def preprocess_data(data, exp_cut=5):

    data = data.copy()
    data = data + 1
    data /= np.mean(data, axis=0)
    vmin, vmax = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data = np.clip(data, vmin, vmax)
    data = (data - data.min()) / (data.max() - data.min())
    return data


def predict(model, data, prob=0.6):
    model.eval()
    inputs = torch.from_numpy(data[:, np.newaxis, :, :]).float().to(device)
    with torch.inference_mode():
        predict_res = model(inputs)
        predict_res = predict_res.softmax(dim=1)[:, 1].cpu().numpy()
        blocks = np.where(predict_res >= prob)[0]
    return blocks


def plot_burst(data, filename, block, offset, file_info, output_dir):
    base_name = os.path.basename(os.path.splitext(filename)[0])
    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)
    load = DataLoader(filename)
    load.load_header()
    file_tstart = load.tstart
    time_reso, freq_reso, tstart, _, freq = file_info
    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    block_tstart   = (block * block_size) * time_reso * tdownsamp + offset
    peak_time    = block_tstart + np.argmax(profile) * time_reso * tdownsamp
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
    plt.xticks(np.linspace(0, w, 6), np.round(block_tstart + np.arange(6) * time_reso * block_size / 5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    output_basename = os.path.join(output_dir, f'{base_name}-{all_time:.4f}-{peak_time:.4f}')
    plt.savefig(f'{output_basename}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    np.save(f'{output_basename}.npy', data)                

    return None


def data_generator(file_list, chunk_size, dds_size, tdownsamp, freq_reso):
    """
    A generator responsible for loading, concatenating, downsampling, and producing data blocks as needed.
    It handles file boundaries and prepares data with overlapping regions for dedispersion.

    Args:
        file_list (list): List of file paths to process.
        chunk_size (int): Target length of each processing block (before downsampling).
        dds_size (int): Additional overlapping data length required for dedispersion (before downsampling).
        tdownsamp (int): Time downsampling factor.
        freq_reso (int): Number of frequency channels.

    Yields:
        tuple: (Current file index, data block for processing)
    """
    buffer = np.array([], dtype=np.float32).reshape(0, freq_reso)
    
    file_idx = 0
    pointer = 0

    while file_idx < len(file_list):
        file_pointer = file_idx
        loader = DataLoader(file_list[file_pointer])
        loader.load_header()

        # Check if the remaining data in the current file is enough for a chunk
        if pointer + chunk_size <= loader.file_len:
            raw_data = loader.load(pointer, chunk_size)
            pointer += chunk_size
        else:
            # Not enough data in the current file, need to read from the next one
            part1_len = loader.file_len - pointer
            raw_data = loader.load(pointer, part1_len)
            file_idx += 1
            pointer = 0

            if file_idx < len(file_list):
                part2_len = chunk_size - part1_len
                next_loader = DataLoader(file_list[file_idx])
                next_loader.load_header()
                
                # Assuming the next file is long enough to provide the remaining data.
                next_data = next_loader.load(0, part2_len)
                raw_data = np.vstack([raw_data, next_data])
                pointer = part2_len
        

        if raw_data.size == 0:
            # This can happen if we are at the very end of the file list
            break

        # Downsampling
        ds_len = raw_data.shape[0] // tdownsamp
        if ds_len == 0: continue
        
        data = np.mean(raw_data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, 
                        raw_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        
        # Add the downsampled data to the buffer
        buffer = np.vstack([buffer, data])
        ds_chunk = chunk_size // tdownsamp
        ds_dds = dds_size // tdownsamp
        # Produce a data block with overlap when the buffer is large enough
        while buffer.shape[0] >= ds_chunk + ds_dds:
            yield file_pointer, buffer[:ds_chunk + ds_dds, :]

            # Remove the produced data from the buffer
            buffer = buffer[ds_chunk:, :]

    # Handle remaining data in the buffer at the end of the file list
    if buffer.shape[0] > 0:
        final_len = ds_chunk + ds_dds
        if buffer.shape[0] < final_len:
            pad_width = final_len - buffer.shape[0]
            padding = np.zeros((pad_width, buffer.shape[1]), dtype=buffer.dtype)
            out_buffer = np.vstack([buffer, padding])
        yield len(file_list) - 1, out_buffer


if __name__ == "__main__":
    args = get_args()
    DM = args.dm
    data_path = args.input
    save_path = args.output
    prob = args.prob
    file_list = handle_regular(data_path, args.re)
    print(f"Will process {len(file_list)} files.")
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    loader = DataLoader(file_list[0])
    file_info = loader.get_params()
    time_reso, freq_reso, _, file_len, freq = file_info
    dds  = (4148808.0 * DM * (freq**-2 - freq.max()**-2) /1000 /time_reso).astype(np.int64)
    dds_size = int(dds.max())
    # Define parameters for the data generator
    chunk_size = 1024 * 512  # Raw samples to read in each step
    ds_chunk = chunk_size // tdownsamp
    ds_dds = (dds // tdownsamp).astype(np.int64)
    # Create the generator
    data_gen = data_generator(file_list, chunk_size, dds_size, tdownsamp, freq_reso)
    model = BinaryNet(base_model, num_classes=2).to(device)
    # model = SPPResNet(base_model, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()


                
    for chunk_idx, (file_idx, data_chunk) in enumerate(data_gen):
        data = data_chunk
        total_chunk = np.ceil((len(file_list) * file_len) / chunk_size).astype(int)
        ### read data
        basename, ext = os.path.splitext(file_list[file_idx])
        file_name = file_list[file_idx]
        print(f"{chunk_idx+1}/{total_chunk}, file: {file_name}")
        t, f = data.shape
        data = np.ascontiguousarray(data, dtype=np.float32)
        ds_dds = np.ascontiguousarray(ds_dds, dtype=np.int64)
        new_data = dedisperse(data, ds_dds, ds_chunk)
        data = new_data
        t, f = data.shape
        if f % 512:
            pad_width  = 512 - (f % 512)
            pad_array = np.zeros((t, pad_width))
            data = np.hstack([data, pad_array])
        t, f = data.shape
        data = np.mean(data.reshape(t//512, 512, 512, f//512), axis=3)
        
        ### predict
        for j in track(range(data.shape[0]), description="Processing..."):
            data[j, :, :] = preprocess_data(data[j, :, :])
        blocks = predict(model, data, prob)
        print(f"Find {len(blocks)} candidates in file {file_name}, chunk idx {chunk_idx}.")
        save_name = basename
        for block in blocks:
            offset = chunk_idx * chunk_size * time_reso
            plotres = plot_burst(data[block], file_name, block, offset, file_info, save_path)
            
                # Ensure we close any figure that was created
    end_time = datetime.now()
    print(f"Total processing time: {(end_time - strt_time).seconds} s")