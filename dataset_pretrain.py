import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import scale,MinMaxScaler
from scipy import signal as sig_fun
df = pd.read_excel("./data_path_pretrain.xlsx") 
pretrain_df, validation_df, test_df = df[df["split"]=="pretrain"], df[df["split"]=="validation"], df[df["split"]=="test"]
class MyDataset(Dataset):
    def __init__(self,path,df,data_type):
        super(MyDataset,self).__init__()
        self.path = path
        self.data_type = data_type
        self.file_list = df
        self.file_list.reset_index(drop=True,inplace=True)

    def __getitem__(self,idx):
        #rhythm: multi-period ECG; median: single-period ECG
        rhythm_path = self.path + "csv_rhythm_all/" + self.file_list.loc[idx,"path"].replace("xml","csv")
        median_path = self.path + "csv_median_all/" + self.file_list.loc[idx,"path"].replace("xml","csv")
        
        def preprocess(file_path):
            df = pd.read_csv(file_path)
            df_lead = df[["I","II","V1","V2","V3","V4","V5","V6"]] 
            # df_lead = df[["I","II"]] 
            sig = df_lead.values.T
            if self.data_type == "train":
                b,a = sig_fun.butter(1,0.5/250,"highpass")
                sig = sig_fun.filtfilt(b,a,sig,axis=1)
                b,a = sig_fun.butter(8,49/250,"lowpass")
                sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
                sig = sig_fun.medfilt(sig,(1,3))
            else:
                b,a = sig_fun.butter(1,0.5/250,"highpass")
                sig = sig_fun.filtfilt(b,a,sig,axis=1)
                b,a = sig_fun.butter(8,35/250,"lowpass")
                sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
                sig = sig_fun.medfilt(sig,(1,3))
                sig = scale(sig,axis=1)
                # sig = sig[0:1,:]  #I lead
            return sig
        rhythm = preprocess(rhythm_path)
        median = preprocess(median_path)
        
        feat = []
        # gender = self.file_list.loc[idx,"SEX"]
        # age =  self.file_list.loc[idx,"AGE"]/100
        # feat +=  [gender,age] 
        rr_mean = (self.file_list.loc[idx,"rr_mean"]-409.7)/70.3
        rr_std =  (self.file_list.loc[idx,"rr_std"]-55.8)/44.0
        feat +=  [rr_mean,rr_std] 
    
        label = abs(self.file_list.loc[idx,"ZhenfaOrdAf"])  
    
        return torch.Tensor(rhythm),torch.Tensor(median),torch.Tensor(feat),label
    def __len__(self):
        return len(self.file_list)

path = "./data/pretrain_csv/"
pretrain_set = MyDataset(path,pretrain_df,"train") 
valid_set = MyDataset(path,validation_df,"valid")
test_set = MyDataset(path,test_df,"test")

