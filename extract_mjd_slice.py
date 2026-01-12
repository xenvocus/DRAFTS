import os

# Keep numba cache writable (same pattern as resnet_search.py)
if "NUMBA_CACHE_DIR" not in os.environ:
    numba_cache_dir = os.path.join(os.getcwd(), "numba_cache")
    os.makedirs(numba_cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir

import argparse

import numpy as np

from DataProc import DataLoader
from DataProc.utils import dedisperse, preprocess_data, load_mask


def _handle_single_file(file_path):
    if not file_path:
        return np.array([])
    return np.array([os.path.abspath(file_path)])


def _auto_tdownsamp(time_reso):
    """Match resnet_search.py heuristic: choose power-of-two so that dt ~= 0.4 ms."""
    # Guard for edge cases
    if time_reso <= 0:
        return 1
    al = int(np.log2(0.4e-3 / time_reso))
    tdownsamp = 2 ** al
    return max(1, int(tdownsamp))


def _compute_dds(dm, freq_mhz, time_reso):
    """Match resnet_search.py formula (in raw sample bins)."""
    dds = (4148808.0 * dm * (freq_mhz ** -2 - freq_mhz.max() ** -2) / 1000.0
        / time_reso
    ).astype(np.int64)
    return dds


def _read_raw_across_files(file_list, start_mjd, raw_len):
    """Read raw (time, pol, freq) samples spanning files.

    Returns:
        raw_data: np.ndarray[time, pol, freq]
        first_file: str
        first_file_tstart: float
        first_file_time_reso: float
        first_file_freq: np.ndarray
    """
    if raw_len <= 0:
        raise ValueError("raw_len must be > 0")

    if len(file_list) == 0:
        raise ValueError("Empty file_list")

    # Find the first file that contains start_mjd (or the earliest one after it)
    start_file_idx = None
    start_sample = 0
    first_meta = None

    for idx, fp in enumerate(file_list):
        loader = DataLoader(fp)
        time_reso, freq_reso, tstart, file_len, freq = loader.get_params()
        duration_days = (file_len * time_reso) / 86400.0
        tend = tstart + duration_days
        # If start_mjd falls before the end of this file, use it.
        if start_mjd < tend:
            start_file_idx = idx
            delta_sec = (start_mjd - tstart) * 86400.0
            start_sample = int(np.floor(delta_sec / time_reso))
            if start_sample < 0:
                start_sample = 0
            if start_sample >= file_len:
                # Shouldn\'t happen given start_mjd < tend, but be defensive.
                start_sample = max(0, file_len - 1)
            first_meta = (fp, tstart, time_reso, freq_reso, freq, file_len)
            break

    if start_file_idx is None:
        # start_mjd is after all files; fall back to last file tail.
        last = DataLoader(file_list[-1])
        time_reso, freq_reso, tstart, file_len, freq = last.get_params()
        first_meta = (file_list[-1], tstart, time_reso, freq_reso, freq, file_len)
        start_file_idx = len(file_list) - 1
        start_sample = max(0, file_len - raw_len)

    first_file, first_tstart, time_reso0, freq_reso0, freq0, _file_len0 = first_meta

    parts = []
    remaining = raw_len
    cur_idx = start_file_idx
    cur_start = start_sample

    while remaining > 0 and cur_idx < len(file_list):
        loader = DataLoader(file_list[cur_idx])
        loader.load_header()
        if abs(loader.time_reso - time_reso0) > 1e-12 or loader.freq_reso != freq_reso0:
            raise ValueError(
                "Inconsistent time_reso/freq_reso across files; cannot safely slice. "
                f"First: dt={time_reso0}, nchans={freq_reso0}; "
                f"This: dt={loader.time_reso}, nchans={loader.freq_reso}"
            )

        can_take = max(0, loader.file_len - cur_start)
        if can_take <= 0:
            cur_idx += 1
            cur_start = 0
            continue

        take = min(remaining, can_take)
        part = loader.load(cur_start, take)
        parts.append(part)
        remaining -= take

        cur_idx += 1
        cur_start = 0

    if len(parts) == 0:
        raise RuntimeError("Failed to read any data from file_list")

    raw_data = np.vstack(parts)

    if remaining > 0:
        # Pad by wrapping existing data (same spirit as data_generator)
        pad_source = raw_data
        while pad_source.shape[0] < remaining:
            pad_source = np.vstack([pad_source, pad_source])
        raw_data = np.vstack([raw_data, pad_source[:remaining]])

    return raw_data, first_file, first_tstart, time_reso0, freq0


def extract_slice_512(
    file_list,
    center_mjd,
    dm,
    tdownsamp,
    mask_file=None,
    exp_cut=5,
):
    """Extract one preprocessed (512,512) slice centered at center_mjd.

    Pipeline matches DRAFTS/resnet_search.py:
    - read raw data
    - time downsample by mean (and average pols)
    - dedisperse using ds_dds
    - take a 1:1 window (time_len = n_freq)
    - adaptive_avg_pool2d to (512,512)
    - apply mask (mapped to 512 columns)
    - preprocess_data

    Returns:
        img_512: np.ndarray[512,512] float32 in [0,1]
    """
    if len(file_list) == 0:
        raise ValueError("file_list is empty")

    # Header from first file for parameters
    loader0 = DataLoader(file_list[0])
    time_reso, freq_reso, _tstart, _file_len, freq = loader0.get_params()

    if tdownsamp <= 0:
        tdownsamp = _auto_tdownsamp(time_reso)

    dds = _compute_dds(dm, freq, time_reso)
    dds_size = int(dds.max())

    ds_dds = (dds // tdownsamp).astype(np.int64)
    ds_max = int(ds_dds.max())

    # In model slicing, block_len equals n_freq (to keep 1:1 aspect)
    block_len_ds = int(freq_reso)

    # Physical duration covered by one block (seconds)
    dt_ds = time_reso * tdownsamp
    duration_sec = block_len_ds * dt_ds

    # Determine block start time so that center_mjd is centered
    start_mjd = center_mjd - (duration_sec / 2.0) / 86400.0

    # We need enough samples for: downsample -> (block_len_ds + ds_max) time bins
    raw_len = int((block_len_ds + ds_max) * tdownsamp)

    raw_data, first_file, first_file_tstart, time_reso0, freq0 = _read_raw_across_files(
        file_list=file_list,
        start_mjd=start_mjd,
        raw_len=raw_len,
    )

    # Downsample by mean over time bins and pol dimension
    ds_len = raw_data.shape[0] // tdownsamp
    if ds_len < block_len_ds + ds_max:
        raise RuntimeError(
            f"Not enough data after downsampling: ds_len={ds_len}, need>={block_len_ds + ds_max}"
        )

    # raw_data: (time, pol, freq)
    # -> data_ds: (ds_len, freq)
    data_ds = (
        np.mean(
            raw_data[: ds_len * tdownsamp].reshape(ds_len, tdownsamp, raw_data.shape[1], freq_reso),
            axis=(1, 2),
        )
    ).astype(np.float32)

    # Dedisperse (output length is exactly block_len_ds)
    new_data = dedisperse(data_ds, ds_dds, block_len_ds, use_numba=True)

    # Resize to (512,512) via adaptive_avg_pool2d (mean pooling)
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(new_data).to(torch.float32).unsqueeze(0).unsqueeze(0)
        img_512 = F.adaptive_avg_pool2d(t, (512, 512)).squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            "Failed to run torch adaptive_avg_pool2d (required to match model sampling). "
            f"Original error: {e}"
        )

    # Apply mask if provided (map raw channel indices -> 512 columns)
    mask_block_indices = None
    if mask_file:
        mask_chans = load_mask(mask_file)
        if mask_chans is not None and len(mask_chans) > 0:
            factor = freq_reso / 512.0
            mask_block_indices = np.unique((mask_chans / factor).astype(int))
            mask_block_indices = mask_block_indices[(mask_block_indices >= 0) & (mask_block_indices < 512)]

    if mask_block_indices is not None and len(mask_block_indices) > 0:
        img_512[:, mask_block_indices] = 0

    img_512 = preprocess_data(img_512, exp_cut=exp_cut).astype(np.float32)

    return img_512


def get_args():
    p = argparse.ArgumentParser(
        description=(
            "定向提取给定中心 MJD 的模型输入切片：dedisperse + 1:1 window + mean-resample -> (512,512)."
        )
    )
    p.add_argument("--mjd", type=float, required=True, help="中心 MJD")
    p.add_argument("-dm", "--dm", type=float, required=True, help="DM")
    p.add_argument(
        "file",
        help="输入数据文件路径（单个文件；不支持通配符/列表）",
    )
    p.add_argument("-o", "--output", type=str, default="./", help="输出目录")
    p.add_argument("-ds", "--tdownsamp", type=int, default=4, help="时间降采样因子；<=0 则自动")
    p.add_argument("--mask", type=str, default=None, help="通道掩膜文件 (raw chan indices)，会映射到 512 列")
    p.add_argument("--exp-cut", type=float, default=5, help="preprocess_data 的 percentile clip 参数")
    return p.parse_args()


def main():
    args = get_args()

    file_list = _handle_single_file(args.file)
    if len(file_list) == 0:
        raise SystemExit("No input files")

    if not os.path.exists(file_list[0]):
        raise SystemExit(f"Input file not found: {file_list[0]}")

    os.makedirs(args.output, exist_ok=True)

    img = extract_slice_512(
        file_list=file_list,
        center_mjd=args.mjd,
        dm=args.dm,
        tdownsamp=args.tdownsamp,
        mask_file=args.mask,
        exp_cut=args.exp_cut,
    )

    base = f"slice_MJD{args.mjd:.9f}_DM{args.dm:g}"
    out_npy = os.path.join(args.output, base + ".npy")
    out_jpg = os.path.join(args.output, base + ".jpg")
    np.save(out_npy, img)

    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    saved_jpg = False
    try:
        from PIL import Image

        Image.fromarray(img_u8, mode="L").save(out_jpg, format="JPEG")
        saved_jpg = True
    except Exception:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.imsave(out_jpg, img_u8, cmap="gray", vmin=0, vmax=255, format="jpg")
            saved_jpg = True
        except Exception as e:
            raise RuntimeError(
                "Failed to save .jpg output. Install Pillow (PIL) or matplotlib. "
                f"Original error: {e}"
            )

    print(f"Saved: {out_npy}")
    if saved_jpg:
        print(f"Saved: {out_jpg}")


if __name__ == "__main__":
    main()
