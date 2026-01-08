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
        self.data_ext = None  # FITS 数据所在的表扩展号（按望远镜映射/自动兜底）


    def load_fil_file(self, start=0, length=None):
        self.load_fil_header()
        fil = FilReader(self.filename)
        self.data = fil.read_block(start, length).astype(np.float32).T
        # 假设没有其他极化
        self.data = self.data.reshape(-1, 1, fil.header.nchans) # [:, :2, :]
        if not self.data.flags['C_CONTIGUOUS']:
            self.data = np.ascontiguousarray(self.data)
        return self.data


    def load_fits_file(self, start=0, length=None):
        """加载由 start 和 length 指定的 FITS 文件部分。"""
        
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
    负责加载、拼接、降采样和按需生成数据块的生成器。
    它处理文件边界并准备具有消色散重叠区域的数据。

    Args:
        file_list (list): 要处理的文件路径列表。
        chunk_size (int): 每个处理块的目标长度 (降采样前)。
        dds_size (int): 消色散所需的额外重叠数据长度 (降采样前)。
        tdownsamp (int): 时间降采样因子。
        freq_reso (int): 频率通道数。
        start_file_idx (int): 起始文件索引。

    Yields:
        tuple: (当前文件索引, 待处理数据块)
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
        # 如果到达列表末尾仍不足 target_raw，则末尾补零 (循环填充 Wrap padding)
        if remaining > 0:
            if raw_data.shape[0] > 0:
                pad_source = raw_data
                while pad_source.shape[0] < remaining:
                    pad_source = np.vstack([pad_source, pad_source])
                pad = pad_source[:remaining]
            else:
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
        # 降采样 Downsampling
        ds_len = raw_data.shape[0] // tdownsamp
        if ds_len == 0: continue
        data = np.mean(raw_data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, 
                        raw_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        # 将降采样后的数据添加到缓冲区
        buffer.extend(data)
        ds_chunk = chunk_size // tdownsamp
        ds_dds = dds_size // tdownsamp
        # 当缓冲区足够大时，生成带有重叠的数据块
        while len(buffer) >= ds_chunk + ds_dds:
            # 从 deque 创建 numpy 数组以进行处理
            out_buffer = np.array(list(itertools.islice(buffer, 0, ds_chunk + ds_dds)))
            yield file_pointer, out_buffer
            # 从左侧弹出，从缓冲区中移除已生成的数据
            for _ in range(ds_chunk):
                buffer.popleft()
    # 在文件列表末尾处理缓冲区中的剩余数据
    if len(buffer) > 0:
        final_len = ds_chunk + ds_dds
        out_buffer = np.array(list(itertools.islice(buffer, 0, len(buffer))))
        if len(buffer) < final_len:
            pad_width = final_len - len(buffer)
            # 从缓冲区本身进行循环填充 (Wrap padding)
            buffer_arr = np.array(list(itertools.islice(buffer, 0, len(buffer))))
            if buffer_arr.shape[0] > 0:
                 pad_source = buffer_arr
                 while pad_source.shape[0] < pad_width:
                     pad_source = np.vstack([pad_source, pad_source])
                 padding = pad_source[:pad_width]
            else:
                 padding = np.zeros((pad_width, out_buffer.shape[1]), dtype=out_buffer.dtype)
            out_buffer = np.vstack([out_buffer, padding])
        yield len(file_list) - 1, out_buffer


def file_generator(file_list, dds_size, tdownsamp, freq_reso, ds_chunk):
    """
    逐个文件加载和处理数据的生成器，处理与下一个文件的拼接以进行消色散重叠。
    """
    current_data = DataLoader(file_list[0]).load()
    for i in range(len(file_list)):
        # 完整加载当前文件
        # 如果有下一个文件，加载其开头以行重叠
        if i + 1 < len(file_list):
            next_loader = DataLoader(file_list[i+1])
            # 假设下一个文件足够长，TODO: 处理不够长的情况
            next_data = next_loader.load()
            # 若下一个文件不足 dds_size，则在 overlap 尾部补零/循环
            if next_data.shape[0] >= dds_size:
                overlap = next_data[:dds_size]
            else:
                pad_len = dds_size - next_data.shape[0]
                # 循环填充 Wrap padding
                if next_data.shape[0] > 0:
                     pad_source = next_data
                     while pad_source.shape[0] < pad_len:
                         pad_source = np.vstack([pad_source, pad_source])
                     pad = pad_source[:pad_len]
                     overlap = np.vstack([next_data, pad])
                else:
                    overlap = np.zeros((dds_size, next_data.shape[1], next_data.shape[2]), dtype=next_data.dtype)
                
            combined_data = np.vstack([current_data, overlap])
            current_data = next_data
        else:
            # 对于最后一个文件，用零填充/循环包裹以保持一致的大小
            # Wrap padding: 将 current_data 开头包裹到结尾
            padshape = ((0, dds_size), (0, 0), (0, 0)) # 默认后备
            if current_data.shape[0] > 0:
                 pad_source = current_data
                 while pad_source.shape[0] < dds_size:
                     pad_source = np.vstack([pad_source, pad_source])
                 pad = pad_source[:dds_size]
                 combined_data = np.vstack([current_data, pad])
            else:
                 combined_data = np.pad(current_data, padshape, mode='constant', constant_values=0)

        # 保证长度足以覆盖 ds_chunk + ds_dds（原始采样域）
        ds_dds = dds_size // tdownsamp
        target_raw = (ds_chunk + ds_dds) * tdownsamp
        if combined_data.shape[0] < target_raw:
            pad_len = target_raw - combined_data.shape[0]
            if combined_data.shape[0] > 0:
                 pad_source = combined_data
                 while pad_source.shape[0] < pad_len:
                     pad_source = np.vstack([pad_source, pad_source])
                 pad = pad_source[:pad_len]
                 combined_data = np.vstack([combined_data, pad])
            else:
                 padshape = ((0, pad_len), (0, 0), (0, 0))
                 combined_data = np.pad(combined_data, padshape, mode='constant', constant_values=0)

        # 降采样 Downsample
        ds_len = combined_data.shape[0] // tdownsamp
        if ds_len == 0:
            continue
        
        data = np.mean(combined_data[:ds_len * tdownsamp].reshape(ds_len, tdownsamp, 
                        combined_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        
        yield i, data


def preload_worker(file_list, chunk_size, dds_size, tdownsamp, freq_reso, ds_chunk, queue):
    """
    运行 data_generator 并将结果放入队列的工作进程。
    它决定使用基于块的处理还是基于文件的处理。
    """
    try:
        if chunk_size > 0:
            # 使用原始的基于块的生成器
            data_gen = data_generator(file_list, chunk_size, dds_size, tdownsamp, freq_reso)
        else:
            data_gen = file_generator(file_list, dds_size, tdownsamp, freq_reso, ds_chunk)
        for item in data_gen:
            queue.put(item)
    finally:
        queue.put(None)  # 哨兵值指示数据结束


