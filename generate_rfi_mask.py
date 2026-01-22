import argparse
import os
import numpy as np
import fitsio
from glob import glob

def _read_data_as_time_freq(data_array: np.ndarray) -> np.ndarray:
    """Normalize FITS DATA column to a (time, nchan) float32 array.

    Common fitsio shapes:
    - (rows, nsblk, npol, nchan, 1)
    - (rows, nsblk, npol, nchan)
    We collapse rows*nsblk into time, and sum over pol.
    """
    arr = data_array
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 4:
        r, nsblk, npol, nchan = arr.shape
        arr = arr.reshape(r * nsblk, npol, nchan)

    if arr.ndim == 3:
        return arr.sum(axis=1, dtype=np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    raise ValueError(f"Unexpected DATA ndim={arr.ndim}, shape={arr.shape}")


def spectral_kurtosis_from_sums(s1: np.ndarray, s2: np.ndarray, m: int) -> np.ndarray:
    """Compute (excess) spectral kurtosis estimator from S1/S2.

    For power samples P_i (i=1..M):
      S1 = sum(P_i), S2 = sum(P_i^2)

    Unbiased SK estimator under ideal Gaussian voltage / exponential power:
      SK = ((M+1)/(M-1)) * ( M*S2/S1^2 - 1 )

    Returns per-channel SK (float64). Invalid channels (S1<=0 or M<2) are NaN.
    """
    sk = np.full_like(s1, np.nan, dtype=np.float64)
    if m < 2:
        return sk
    # Ensure S1^2 is not zero and S1, S2 are valid
    good = (s1 > 1e-9) & np.isfinite(s1) & np.isfinite(s2)
    s1g = s1[good].astype(np.float64, copy=False)
    s2g = s2[good].astype(np.float64, copy=False)
    
    # Check for potential underflow/division by zero in S1^2
    denom = s1g * s1g
    valid_denom = denom > 1e-18
    
    if not np.any(valid_denom):
        return sk

    # Filter strictly valid denominator
    s1g = s1g[valid_denom]
    s2g = s2g[valid_denom]
    
    # Use indices to update 'good'
    # Actually simpler to just calculate for all 'good' where denom is safe.
    # But for simplicity, let's trust s1 > 1e-9 implies s1^2 > 1e-18.
    
    sk_val = ((m + 1.0) / (m - 1.0)) * ((m * s2g) / (s1g * s1g) - 1.0)
    
    # Map back to full array
    # We need to be careful with indices if we filtered twice.
    # Let's just use the first 'good' mask which ensures s1 > 1e-9.
    sk[good] = sk_val
    return sk


def process_file(filepath, output_dir, sigma_thresh=10.0):
    try:
        # Read Header to get shapes
        h0 = fitsio.read_header(filepath)
        
        # Determine extension. Logic borrowed from DataProc/__init__.py
        tel = None
        try:
             tel = h0['TELESCOP']
        except:
             try: tel = h0['OBSERVAT']
             except: pass
        
        data_ext = 1
        if tel and 'EFFELSBERG' in str(tel).upper():
            data_ext = 2
            
        h = fitsio.read_header(filepath, ext=data_ext)
        nchan = h['NCHAN']
        nsblk = h['NSBLK']
        npol = h.get('NPOL', 1)
        total_rows = h['NAXIS2']

        # Spectral Kurtosis (SK): stream through rows and only accumulate S1 and S2.
        # This avoids large temporary arrays from **3/**4 on the full matrix.
        rows_per_chunk = 8
        n_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk

        s1 = np.zeros(nchan, dtype=np.float64)
        s2 = np.zeros(nchan, dtype=np.float64)
        m = 0

        with fitsio.FITS(filepath) as ff:
            for ci in range(n_chunks):
                start_row = ci * rows_per_chunk
                end_row = min((ci + 1) * rows_per_chunk, total_rows)
                if start_row >= end_row:
                    break

                # Read block. Shape is typically (n_rows, nsblk, npol, nchan)
                # fitsio might squeeze dimensions if they are 1.
                # To be safe, we use reshape based on header info.
                block = ff[data_ext].read(rows=range(start_row, end_row), columns=['DATA'])['DATA']
                
                # Expected number of elements
                current_rows = end_row - start_row
                expected_size = current_rows * nsblk * npol * nchan
                if block.size != expected_size:
                    print(f"Warning: Block size mismatch in {filepath}. Expected {expected_size}, got {block.size}. Skipping chunk.")
                    continue

                # Reshape to (Time_in_chunk, Pol, Chan)
                # Time_in_chunk = current_rows * nsblk
                flat = block.reshape(current_rows * nsblk, npol, nchan)
                
                # Convert to Float32 and Sum Pols -> (Time, Chan)
                if npol > 1:
                    tf = flat.sum(axis=1, dtype=np.float32)
                else:
                    tf = flat.reshape(current_rows * nsblk, nchan).astype(np.float32)

                # S1
                s1 += tf.sum(axis=0, dtype=np.float64)
                # S2: square in-place then sum
                np.square(tf, out=tf)
                s2 += tf.sum(axis=0, dtype=np.float64)
                m += tf.shape[0]

        sk = spectral_kurtosis_from_sums(s1, s2, m)
        good = np.isfinite(sk)
        if not np.any(good):
            return

        med = np.median(sk[good])
        mad = np.median(np.abs(sk[good] - med))
        sigma = 1.4826 * mad
        if sigma == 0 or not np.isfinite(sigma):
            sigma = np.std(sk[good])

        final_mask = np.zeros(nchan, dtype=bool)
        if sigma > 0 and np.isfinite(sigma):
            final_mask[good] = np.abs(sk[good] - med) > sigma_thresh * sigma

        bad_channels = np.where(final_mask)[0]

        # Write to file
        basename = os.path.basename(filepath)
        mask_name = basename.replace('.fits', '.mask') # or .fits.mask
        if output_dir:
            out_path = os.path.join(output_dir, mask_name)
        else:
            out_path = os.path.join(os.path.dirname(filepath), mask_name)
            
        with open(out_path, 'w') as f:
            # Write one channel per line
            for ch in bad_channels:
                f.write(f"{ch}\n")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate RFI mask using Spectral Kurtosis (SK) filter")
    parser.add_argument('input', nargs='+', help="Input FITS files")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output directory")
    parser.add_argument('-t', '--threshold', type=float, default=3.0, help="Sigma threshold (larger -> fewer masked channels)")
    
    args = parser.parse_args()
    
    files = []
    for p in args.input:
        files.extend(glob(p))
    files = sorted(list(set(files)))
    
    if not files:
        print("No files found.")
        return
        
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
        
    for f in files:
        process_file(f, args.output, args.threshold)

if __name__ == "__main__":
    main()
