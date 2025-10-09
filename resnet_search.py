import os, re, sys
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_color_codes()
from sigpyproc.readers import FilReader
import torch, torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
from BinaryClass.binary_model import SPPResNet, BinaryNet


### 读取fits文件，只保留两维数据
def load_fits_file(file_name, reverse_flag=False):
    try:
        import fitsio
        data, h  = fitsio.read(file_name, header=True)
    except:
        with fits.open(file_name) as f:
            h    = f[1].header
            data = f[1].data
    data         = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], h['NPOL'], h['NCHAN'])[:, :2, :]
    # data         = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], int(h['NCHAN']))[:, :]
    # data = data[:, np.newaxis, :]
    # if reverse_flag: data = np.array(data[:, :, ::-1])
    if reverse_flag: data = np.array(data[:, ::-1])
    return data


def load_fil_file(file_name, reverse_flag=False):

    fil = FilReader(file_name)
    data = fil.read_block(0, fil.header.nsamples).astype(np.float32)
    data = data.reshape(fil.header.nsamples, fil.header.nchans)
    data = data[:, np.newaxis, :]
    if reverse_flag: data = np.array(data[:, :, ::-1])
    return data
    


def preprocess_data(data, exp_cut=5):

    data         = data.copy()
    data         = data + 1
    w, h         = data.shape
    data        /= np.mean(data, axis=0)
    vmin, vmax   = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data         = np.clip(data, vmin, vmax)
    data         = (data - data.min()) / (data.max() - data.min())

    return data


def plot_burst(data, filename, block):

    fig          = plt.figure(figsize=(5, 5))
    gs           = gridspec.GridSpec(4, 1)

    w, h         = data.shape
    profile      = np.mean(data, axis=1)
    time_start   = ((fits_number - 1) * file_leng + block * block_size) * time_reso
    peak_time    = time_start + np.argmax(profile) * time_reso

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(gs[0, 0])
    plt.plot(profile, color='royalblue', alpha=0.8, lw=1)
    plt.scatter(np.argmax(profile), np.max(profile), color='red', s=100, marker='x')
    plt.xlim(0, w)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(gs[1:, 0])
    plt.imshow(data.T, origin='lower', cmap='mako', aspect='auto')
    plt.scatter(np.argmax(profile), 0, color='red', s=100, marker='x')
    plt.yticks(np.linspace(0, h, 6), np.int64(np.linspace(freq.min(), freq.max(), 6)))
    plt.xticks(np.linspace(0, w, 6), np.round(time_start + np.arange(6) * time_reso * block_size / 5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig('{}-{:0>4d}-{}.jpg'.format(filename, block, peak_time), format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-dm', '--dm', type=int, default=893)
    args.add_argument('-i', '--input', type=str, default='./')
    args.add_argument('-o', '--output', type=str, default='./')
    args.add_argument('-re', type=str, default='*.fits')
    args = args.parse_args()
    ### path config
    down_sampling_rate        = 8
    DM                        = args.dm
    date_path                 = args.input
    save_path                 = args.output
    prob                      = 0.6
    block_size                = 512
    base_model                = 'resnet18'
    model_path                = './class_resnet18.pth'


    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass
    if "{" in args.re and "}" in args.re:
        re_left = args.re.split("{")[0]
        re_brace = args.re.split("{")[1].split("}")[0]
        re_right = args.re.split("}")[1]
        re_brace = re_brace.replace(", ", ",")
        cntnt = re_brace.split(",")

        file_list = []
        for i in range(len(cntnt)):
            express = re_left + cntnt[i] + re_right
            globi = glob(date_path + express)
            file_list.extend(globi)
    else:
        file_list = glob(date_path + args.re)
    file_list = np.sort(file_list)
    file_list                 = np.append(file_list, file_list[-1])

    ### file params read
    # with fits.open(date_path + file_list[0]) as f:
    if file_list[0].endswith('.fil'):
        fil = FilReader(file_list[0])
        time_reso             = fil.header.tsamp * down_sampling_rate
        freq_reso             = int(fil.header.nchans)
        file_leng             = fil.header.nsamples // down_sampling_rate
        foff                  = fil.header.foff
        fch1                  = fil.header.fch1
        freq                  = fch1 + np.arange(freq_reso) * foff
            
    else:
        with fits.open(file_list[0]) as f:
            time_reso             = f[1].header['TBIN'] * down_sampling_rate
            freq_reso             = int(f[1].header['NCHAN'])
            file_leng             = f[1].header['NAXIS2'] * f[1].header['NSBLK']  // down_sampling_rate
            freq                  = f[1].data['DAT_FREQ'][0, :].astype(np.float64)
    reverse_flag              = False
    if freq[0] > freq[-1]:
        reverse_flag          = True
        freq                  = np.array(freq[::-1])

    ### time delay correct
    dds                       = (4.15 * DM * (freq**-2 - freq.max()**-2) * 1e3 / time_reso).astype(np.int64)
    if file_leng % 512:
        redundancy            = ((file_leng // 512) + 1) * 512 - file_leng
    else:
        redundancy            = 0

    comb_leng                 = int(dds.max() / file_leng) + 1
    comb_file_leng            = (file_leng + redundancy + dds.max()) * down_sampling_rate
    down_file_leng            = file_leng + redundancy

    ### model config
    model                     = BinaryNet(base_model, num_classes=2).to(device)

#    model                     = SPPResNet(base_model, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    ### read data
    for i in range(len(file_list) - 1):
        def load_file(file_name, reverse_flag=reverse_flag):
            if file_name.endswith('.fil'):
                data = load_fil_file(file_name, reverse_flag)
            else:
                data = load_fits_file(file_name, reverse_flag)
            return data
        # raw_data              = load_fits_file(date_path + file_list[i], reverse_flag)
        raw_data              = load_file(file_list[i], reverse_flag)
        # raw_data             = raw_data[:, np.newaxis, :]
        fits_number           = i + 1
        filename              = file_list[i].split('.fits')[0]
        print(filename)

        for j in range(comb_leng):
            if i + j + 1      < len(file_list):
                # raw_data      = np.append(raw_data, load_fits_file(date_path + file_list[i+j+1], reverse_flag), axis=0)
                raw_data      = np.append(raw_data, load_file(file_list[i+j+1], reverse_flag), axis=0)
        if raw_data.shape[0]  < comb_file_leng:
            raw_data          = np.append(raw_data, np.random.rand(comb_file_leng-raw_data.shape[0], raw_data.shape[1], freq_reso) * raw_data.max() / 2, axis=0)
            # raw_data          = np.append(raw_data, np.random.rand(comb_file_leng-raw_data.shape[0], freq_reso) * raw_data.max() / 2, axis=0)
        # raw_data              = raw_data[:comb_file_leng, :, :]
        raw_data              = raw_data[:comb_file_leng, :]
        print(raw_data.shape)
        data                  = np.mean(raw_data.reshape(raw_data.shape[0] // down_sampling_rate, down_sampling_rate, raw_data.shape[1], freq_reso), axis=(1, 2)).astype(np.float32)
        # data                  = np.mean(raw_data.reshape(raw_data.shape[0] // down_sampling_rate, down_sampling_rate, freq_reso), axis=(1, 2)).astype(np.float32)

        new_data              = np.zeros((down_file_leng, freq_reso))
        
        for j in range(freq_reso):
            new_data[:, j]    = data[dds[j]: dds[j]+down_file_leng, j]
        if new_data.shape[1] % 512:
            pad_width        = 512 - (new_data.shape[1] % 512)
            pad_array        = np.zeros((new_data.shape[0], pad_width))
            new_data         = np.hstack([new_data, pad_array])
        if new_data.shape[0] % 512:
            pad_width        = 512 - (new_data.shape[0] % 512)
            pad_array        = np.zeros((pad_width, new_data.shape[1]))
            new_data         = np.vstack([new_data, pad_array])
        data                  = np.mean(new_data.reshape(new_data.shape[0]//512, 512, 512, new_data.shape[1]//512), axis=3)
        print(data.shape)
        ### predict
        for j in range(data.shape[0]):
            data[j, :, :]     = preprocess_data(data[j, :, :])
        inputs                = torch.from_numpy(data[:, np.newaxis, :, :]).float().to(device)
        predict_res           = model(inputs)

        ### plot
        with torch.no_grad():
            predict_res       = predict_res.softmax(dim=1)[:, 1].cpu().numpy()
        blocks                = np.where(predict_res >= prob)[0]
        # save_name = os.path.join(save_path, filename) 
        save_name = filename
        for block in blocks:
            plotres           = plot_burst(data[block], save_name, block)
            np.save('{}-{:0>4d}.npy'.format(save_name, block), data[block])

