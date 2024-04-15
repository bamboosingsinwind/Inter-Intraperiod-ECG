#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.rcParams['xtick.labelsize'] = 14*3*4
plt.rcParams['ytick.labelsize'] = 14*3*4

dir = "/home/zxq/CAD/data/csv_rhythm_all/"
name_list = os.listdir(dir)
path = dir + name_list[1]###0
signal = pd.read_csv(path)["I"].to_numpy()

#============== Original Signal==================
plt.figure(figsize=(50*4, 10*4),dpi=200)

plt.plot(signal,linestyle='-', color='blue', linewidth=2)
# plt.title('Original Signal', fontsize=16*20, fontweight='normal')
plt.xlabel('Time', fontsize=14*20, fontweight='normal')
plt.ylabel('Magnitude', fontsize=14*20, fontweight='normal')
plt.tight_layout()
plt.show()
# plt.savefig("./figures/high/original_rhythm011notitle.pdf",format='pdf')

#============== FFT ==================
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(signal), 1/500)

half_length = len(signal)//2
fft_result = fft_result[:half_length]
fft_freq = fft_freq[:half_length]

plt.figure(figsize=(50*4, 10*4),dpi=200)
plt.plot(fft_freq, np.abs(fft_result),linestyle='-', linewidth=2)# color='green',
# plt.title('Fast Fourier Transform', fontsize=16*20, fontweight='normal')
plt.xlabel('Frequency', fontsize=14*20, fontweight='normal')
plt.ylabel('Magnitude', fontsize=14*20, fontweight='normal')

plt.tight_layout()
plt.show()
# plt.savefig("./figures/high/fft_rhythm011notitle.pdf",format='pdf')

# %%
#============== Denoising ==================
from scipy import signal as sig_fun
df = pd.read_csv(path)
df_lead = df[["I","II","V1","V2","V3","V4","V5","V6"]] #8 leads
sig_origin = df_lead.values.T

sig = sig_origin
b,a = sig_fun.butter(1,0.5/250,"highpass")
sig = sig_fun.filtfilt(b,a,sig,axis=1)
b,a = sig_fun.butter(8,30/250,"lowpass")
sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
sig = sig_fun.medfilt(sig,(1,3))

plt.figure(figsize=(50*4, 10*4),dpi=200)
plt.plot(sig_origin[0,:],linestyle='-', color='blue', linewidth=2,label="original")
plt.plot(sig[0,:],linestyle='-', color='red', linewidth=2,label="denoised")
# plt.title('Signal Denoising', fontsize=16*20, fontweight='normal')
plt.xlabel('Time', fontsize=14*20, fontweight='normal')
plt.ylabel('Magnitude', fontsize=14*20, fontweight='normal')
plt.legend(fontsize=14*3*4)
plt.tight_layout()
plt.show()
# plt.savefig("./figures/high/denoising0notitle.pdf",format='pdf')
# plt.savefig("./figures/denoising0.png")
