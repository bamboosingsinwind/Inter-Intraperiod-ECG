import matplotlib.pyplot as plt
from scipy import signal as sig_fun
import pandas as pd
# path = "D:\\seadrive\\data\\朱向前_1\\我的资料库\\长庚\\10001001_79688458_211130117000161.csv"
path = 'D:/seadrive/data/朱向前_1/我的资料库/长庚/folder_shuanse/XML/10001413_409705_160801117000193.csv'
# df = pd.read_excel(path)
df = pd.read_csv(path)
df_lead = df[["I","II","V1","V2","V3","V4","V5","V6"]] #8 leads
sig_origin = df_lead.values.T

sig = sig_origin
b,a = sig_fun.butter(1,0.5/250,"highpass")
sig = sig_fun.filtfilt(b,a,sig,axis=1)
b,a = sig_fun.butter(8,30/250,"lowpass")
sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
sig = sig_fun.medfilt(sig,(1,3))

plt.figure(figsize=(400//2, 50),dpi=200)
plt.plot(sig_origin[0,500-50:1500-50],linestyle='-', color='blue', linewidth=2,label="original")
plt.plot(sig[0,500-50:1500-50],linestyle='-', color='red', linewidth=2,label="denoised")
# plt.title('Signal Denoising', fontsize=16*20, fontweight='normal')
plt.xlabel('Samples', fontsize=10*8*3, fontweight='normal')
plt.ylabel('Magnitude', fontsize=10*8*3, fontweight='normal')
plt.legend(fontsize=8*8*3, loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=8*8*3)
plt.tight_layout()
# plt.show()
plt.savefig("C:/Users/26284/Desktop/denoising0notitle_tl2_0921.pdf",format='pdf')