import os
import fitsio
import itertools
import numpy as np
from collections import deque
from sigpyproc.readers import FilReader


class DataLoader:
    def __init__(self, filename, telescope='Fake', backend='Fake'):
        self.filename = filename
        self.telescope = telescope
        self.backend = backend
        self.data_ext = None  # FITS 数据所在的表扩展号（按望远镜映射/自动兜底）


    def load_fil_file(self, start=0, length=None):
        self.load_fil_header()
        fil = FilReader(self.filename)
        self.data = fil.read_block(start, length).astype(np.float32).T
        # assume no other pols
        self.data = self.data.reshape(-1, 1, fil.header.nchans) # [:, :2, :]
        if not self.data.flags['C_CONTIGUOUS']:
            self.data = np.ascontiguousarray(self.data)
        return self.data


    def load_fits_file(self, start=0, length=None):
        """Load a portion of the FITS file specified by start and length."""
        
        if start == 0 and length is None:
            # 统一依据 self.data_ext 读取
            self.load_fits_header()
            data, h  = fitsio.read(self.filename, header=True, ext=self.data_ext)
            data = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], h['NPOL'], h['NCHAN'])[:, :2, :]
            self._reverse()
        else: 
            self.load_fits_header()
            nsblk = self.header['NSBLK']
            start_nsub = int(start/nsblk) 
            start_nsamp = start - start_nsub * nsblk
            end_nsub = np.ceil((start + length) / nsblk).astype(int) 
            end_nsamp = start_nsamp + length
            sub_idcs = np.arange(start_nsub, end_nsub)
            data, h = fitsio.read(self.filename, rows=sub_idcs,
                                    columns=['DATA'], ext=self.data_ext, header=True)
            data = data['DATA'].reshape(-1, h['NPOL'], h['NCHAN'])[start_nsamp:end_nsamp, :2, :]
        if not data.flags['C_CONTIGUOUS']:
            self.data = np.ascontiguousarray(data)
        else:
            self.data = data
        return self.data


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
        # 主头
        h0 = fitsio.read_header(self.filename)
        # 基于望远镜硬编码映射选择数据扩展号
        # 注意：不写外部配置，直接在此硬编码映射
        telescope_profiles = {
            'EFFELSBERG': {'fits_data_ext': 2, 'data_column': 'DATA'},
        }
        # 获取望远镜名称（全部转大写用于匹配）；若无则置空
        tel = None
        try:
            tel = h0['TELESCOP']
        except Exception:
            try:
                tel = h0['OBSERVAT']
            except Exception:
                tel = None
        tel_key = str(tel).upper() if tel is not None else ''
        # 默认扩展 1
        data_ext = 1
        # 简单包含匹配（如 'EFFELSBERG'）
        for k, cfg in telescope_profiles.items():
            if k in tel_key:
                data_ext = cfg.get('fits_data_ext', 1)
                break
        self.data_ext = data_ext
        # 读取目标扩展头
        h = fitsio.read_header(self.filename, ext=self.data_ext)
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


def data_generator(file_list, chunk_size, dds_size, tdownsamp, freq_reso, start_file_idx=0):
    """
    A generator responsible for loading, concatenating, downsampling, 
    and producing data blocks as needed. It handles file boundaries and 
    prepares data with overlapping regions for dedispersion.

    Args:
        file_list (list): List of file paths to process.
        chunk_size (int): Target length of each processing block (before downsampling).
        dds_size (int): Additional overlapping data length required for dedispersion 
        (before downsampling).
        tdownsamp (int): Time downsampling factor.
        freq_reso (int): Number of frequency channels.
        start_file_idx (int): Starting file index.

    Yields:
        tuple: (Current file index, data block for processing)
    """
    buffer = deque()

    file_idx = start_file_idx
    pointer = 0

    # 基于首文件估算一次跨文件深度（用于控制预期，逻辑上可超出）
    if len(file_list) > 0:
        _tmp_loader0 = DataLoader(file_list[0])
        _tmp_loader0.load_header()
        first_len = max(1, _tmp_loader0.file_len)
    else:
        first_len = 1
    # 目标原始长度需要包含 dedispersion 的重叠区，否则末端 shift 会越界
    target_raw = chunk_size + dds_size
    est_files_per_window = int(np.ceil(target_raw / first_len))

    while file_idx < len(file_list):
        file_pointer = file_idx
        loader = DataLoader(file_list[file_pointer])
        loader.load_header()

        # 从当前文件开始，累积读取直至达到 target_raw 或到达列表末尾
        raw_parts = []
        remaining = target_raw
        crosses = 0
        cur_idx = file_idx
        cur_ptr = pointer
        while remaining > 0 and cur_idx < len(file_list):
            cur_loader = DataLoader(file_list[cur_idx])
            cur_loader.load_header()
            can_take = max(0, cur_loader.file_len - cur_ptr)
            if can_take > 0:
                take = min(remaining, can_take)
                part = cur_loader.load(cur_ptr, take)
                raw_parts.append(part)
                remaining -= take
                cur_ptr += take
                # 若当前文件消耗完，则切至下一个
                if cur_ptr >= cur_loader.file_len:
                    cur_idx += 1
                    cur_ptr = 0
            else:
                # 当前文件无可读，直接跳到下一文件
                cur_idx += 1
                cur_ptr = 0
            crosses += 1
            # 若跨越次数已达到预估深度且仍需更多数据，继续尝试下一个文件；
            # 到列表末尾后会退出循环，后续统一补零。
            if crosses >= est_files_per_window and remaining <= 0:
                break

        if len(raw_parts) == 0:
            # 文件已耗尽
            break

        raw_data = np.vstack(raw_parts)
        # 如果到达列表末尾仍不足 target_raw，则末尾补零
        if remaining > 0:
            pad = np.zeros((remaining, raw_data.shape[1], raw_data.shape[2]), dtype=raw_data.dtype)
            raw_data = np.vstack([raw_data, pad])
            # 更新到列表尾部状态
            file_idx = len(file_list)  # 触发后续结束
            pointer = 0
        else:
            # 更新到最后一次读取位置
            file_idx = cur_idx
            pointer = cur_ptr

        if raw_data.size == 0:
            break
        # Downsampling
        ds_len = raw_data.shape[0] // tdownsamp
        if ds_len == 0: continue
        data = np.mean(raw_data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, 
                        raw_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        # Add the downsampled data to the buffer
        buffer.extend(data)
        ds_chunk = chunk_size // tdownsamp
        ds_dds = dds_size // tdownsamp
        # Produce a data block with overlap when the buffer is large enough
        while len(buffer) >= ds_chunk + ds_dds:
            # Create a numpy array from the deque for processing
            out_buffer = np.array(list(itertools.islice(buffer, 0, ds_chunk + ds_dds)))
            yield file_pointer, out_buffer
            # Remove the produced data from the buffer by popping from the left
            for _ in range(ds_chunk):
                buffer.popleft()
    # Handle remaining data in the buffer at the end of the file list
    if len(buffer) > 0:
        final_len = ds_chunk + ds_dds
        out_buffer = np.array(list(itertools.islice(buffer, 0, len(buffer))))
        if len(buffer) < final_len:
            pad_width = final_len - len(buffer)
            padding = np.zeros((pad_width, out_buffer.shape[1]), dtype=out_buffer.dtype)
            out_buffer = np.vstack([out_buffer, padding])
        yield len(file_list) - 1, out_buffer


def file_generator(file_list, dds_size, tdownsamp, freq_reso, ds_chunk):
    """
    A generator that loads and processes data file by file, handling concatenation
    with the next file for dedispersion overlap.
    """
    current_data = DataLoader(file_list[0]).load()
    for i in range(len(file_list)):
        # Load current file completely
        # If there is a next file, load the beginning of it for overlap
        if i + 1 < len(file_list):
            next_loader = DataLoader(file_list[i+1])
            # Here we assume next file is long enough, TODO: handle case if not enough
            next_data = next_loader.load()
            # 若下一个文件不足 dds_size，则在 overlap 尾部补零
            if next_data.shape[0] >= dds_size:
                overlap = next_data[:dds_size]
            else:
                pad_len = dds_size - next_data.shape[0]
                padshape = ((pad_len, 0), (0, 0), (0, 0))
                overlap = np.pad(next_data, padshape, mode='constant', constant_values=0)[:dds_size]
            combined_data = np.vstack([current_data, overlap])
            current_data = next_data
        else:
            # For the last file, pad with zeros to maintain consistent size
            padshape = ((0, dds_size), (0, 0), (0, 0))
            combined_data = np.pad(current_data, padshape, mode='constant', constant_values=0)

        # 保证长度足以覆盖 ds_chunk + ds_dds（原始采样域）
        ds_dds = dds_size // tdownsamp
        target_raw = (ds_chunk + ds_dds) * tdownsamp
        if combined_data.shape[0] < target_raw:
            pad_len = target_raw - combined_data.shape[0]
            padshape = ((0, pad_len), (0, 0), (0, 0))
            combined_data = np.pad(combined_data, padshape, mode='constant', constant_values=0)

        # Downsample
        ds_len = combined_data.shape[0] // tdownsamp
        if ds_len == 0:
            continue
        
        data = np.mean(combined_data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, 
                        combined_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        
        yield i, data


def preload_worker(file_list, chunk_size, dds_size, tdownsamp, freq_reso, ds_chunk, queue):
    """
    A worker process that runs the data_generator and puts the results in a queue.
    It decides whether to use chunk-based or file-based processing.
    """
    try:
        if chunk_size > 0:
            # Use the original chunk-based generator
            data_gen = data_generator(file_list, chunk_size, dds_size, tdownsamp, freq_reso)
        else:
            data_gen = file_generator(file_list, dds_size, tdownsamp, freq_reso, ds_chunk)
        for item in data_gen:
            queue.put(item)
    finally:
        queue.put(None)  # Sentinel value to indicate the end of data


