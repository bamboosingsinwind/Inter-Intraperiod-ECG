import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import scale,MinMaxScaler
from scipy import signal as sig_fun
from sklearn.model_selection import train_test_split

df = pd.read_excel("./data_path_finetune.xlsx")
train_df, validation_df, test_df = df[df["split"]=="train"], df[df["split"]=="validation"], df[df["split"]=="test"]
print(train_df.shape)
print(test_df)
class MyDataset(Dataset):
   
    def __init__(self,path,df,data_type):
        super(MyDataset,self).__init__()
        self.path = path
        self.data_type = data_type
        self.file_list = df
        self.file_list.reset_index(drop=True,inplace=True)
       
    def __getitem__(self,idx):
        file_path = self.path + self.file_list.loc[idx,"path"].replace("xml","csv")
        feat = []
        gender = self.file_list.loc[idx,"SEX"]
        age =  self.file_list.loc[idx,"AGE"]/100
        feat +=  [gender,age] #[0,0]
        
        feat_plus = torch.FloatTensor(feat)
        df = pd.read_csv(file_path)
        df_lead = df[["I","II","V1","V2","V3","V4","V5","V6"]] #8 leads
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
        lead = torch.FloatTensor(np.copy(sig))
        label = abs(self.file_list.loc[idx,"ZhenfaOrdAf"])  
       
        return lead,feat_plus,label
    
    def __len__(self):
        return len(self.file_list)
path = "./data/csv_rhythm_all/" 
# path = "./data/csv_median_all/"

train_set = MyDataset(path,train_df,"train") 
valid_set = MyDataset(path,validation_df,"valid")
test_set = MyDataset(path,test_df,"test")

print(len(train_set))
print(len(valid_set))
print(len(test_set))

