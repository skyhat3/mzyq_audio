import os
import re
import torch
import torchaudio
import librosa
import pandas as pd
import numpy as np
import torchaudio.functional as F

from pydub import AudioSegment
from pydub.utils import make_chunks
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB,MFCC
from torch.utils.data import Dataset

def out_csv(dir,name='<CSV>'):
    classes=os.listdir(dir)
    instruments=[]
    ids=[]
    labels=[]
    filenames=[]
    for label,sub_dir in enumerate(classes):
        for filename in os.listdir(os.path.join(dir,sub_dir)):
            filenames.append(filename)
            instrument,id=sub_dir[:-5],sub_dir[-5:]
            instruments.append(instrument)
            ids.append(id)
            labels.append(label)
    frame=pd.DataFrame(columns=['id','ins_name','filename','label'])
    frame['id']=ids
    frame['ins_name']=instruments
    frame['filename']=filenames
    frame['label']=labels
    frame.to_csv(name)

def open_audio(filename,freq=32000):
    try:
        waveform, sr = torchaudio.load(filename, channels_first=True,normalize=True)
        waveform=F.resample(waveform,sr,new_freq=freq)
        return waveform, freq
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None, None

def pre_process(waveform,sr):
    pipeline = torch.nn.Sequential(
        MelSpectrogram(sample_rate=sr, n_mels=32),  # 计算梅尔频谱图
        AmplitudeToDB()  # 将梅尔频谱图转换为分贝表示
    )
    specgram = pipeline(waveform)
    return specgram


class CustomDataset(Dataset):
    def __init__(self,csv_file,transforms=None,dir=r'.\4.民族乐器音色打分数据库\分割音频'):
        self.dir=dir
        self.data_frame=pd.read_csv(csv_file)
        self.transforms=transforms
        self.labels=self.data_frame['label'].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        filename=self.data_frame['filename'][index]
        sub_dir=self.data_frame['ins_name'][index]+self.data_frame['id'][index]
        filename=os.path.join(self.dir,sub_dir,filename)
        audio,sr=open_audio(filename)
        # print(audio.shape,sr) # torch.Size([1, 448000]) 32000
        audio=pre_process(audio,sr)
        if audio.dim()==3:
            _,c,n=audio.size()
            audio=audio.view(c,n)
        return audio,self.labels[index]
    
    def get_labels(self):
        return self.labels
    

class NumpyDataset(Dataset):
    def __init__(self, audio_path='audio_data.npy', labels_path='labels.npy', transforms=None):
        self.audio_data = np.load(audio_path, allow_pickle=True)
        self.labels = np.load(labels_path)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        audio = self.audio_data[index]
        label = self.labels[index]
        
        audio_tensor = torch.tensor(audio)
        label_tensor = torch.tensor(label)
        
        if self.transforms is not None:
            audio_tensor = self.transforms(audio_tensor)
        
        return audio_tensor, label_tensor

def save_npy(dataset):
    audio_data = []
    labels = []
    for i in range(len(dataset)):
        audio, label = dataset[i]
        audio_data.append(audio.numpy())  # 将 PyTorch 张量转换为 NumPy 数组
        labels.append(label)

    # 将音频数据和标签保存为 .npy 文件
    np.save('audio_data.npy', np.stack(audio_data), allow_pickle=True)  # 保存音频数据
    np.save('labels.npy', np.array(labels), allow_pickle=True)          # 保存标签
    # 检查保存的数据
    print("音频数据已保存为 'audio_data.npy'，标签已保存为 'labels.npy'")


def convert_to_model_input(filtered_audio, target_length=88200):
        if len(filtered_audio) < target_length:
            return np.pad(filtered_audio, (0, target_length - len(filtered_audio)))
        else:
            return filtered_audio[:target_length]
        
if __name__=='__main__':
    target_dir=r'.\4.民族乐器音色打分数据库\分割音频'
    csv_name='yqlb.csv'
    # out_csv(target_dir,name=csv_name)
    frame=pd.read_csv(csv_name)
    
    dataset=CustomDataset(csv_name,dir=target_dir)
    save_npy(dataset)
    audio_file='audio_data.npy'
    label_file='labels.npy'
    # dataset=NumpyDataset(audio_file,label_file)
    # print(dataset[0][0].shape)


    # metadata = pd.read_csv(r'C:\mzyq\UrbanSound8K\metadata\UrbanSound8K.csv')
    
    # # 创建数据集实例
    # dataset = UrbanSound8KDataset(r'C:\mzyq\UrbanSound8K\audio', metadata)
    # print(dataset[0][0].shape)