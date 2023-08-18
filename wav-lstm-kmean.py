#编写一个程序，能够读取文件夹audio内的mp3文件，采用自监督的方式对文件进行分类，可以采用torch、wav2vec、sklearn进行特征提取与分类，分类完成后需要对文件夹内文件的分类结果进行绘图展示，将类别标签保存为csv

import os
import csv
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils

device = torch.device("cuda:0")

# 加载wav2vec模型
def load_wav2vec_model(model_path):
    
    # cp = torch.load(model_path)
    # model = Wav2VecModel.build_model(cp['args'], task=None)
    # model.load_state_dict(cp['model'])
    # model.eval()
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("loading model(s) from {}".format(model_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",)

    model = models[0]
    model = model#.half()
    model = model.to(device)
    
    #model.to('cpu').float()
    model.eval()
    
    print("eval ")
    return model

# 加载音频文件
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    return waveform

# 提取音频特征
def extract_features(wav2vec_model, audio_path):
    audio = load_audio(audio_path).to(device)#.half()
    print("extract_features")
    features = wav2vec_model.feature_extractor(audio)
    
    # torch.Size([2, 512, 18647])
    features = features.transpose(1, 2).reshape(-1, 2*512).unsqueeze(0)
   
    # 定义LSTM模型
    lstm_model = nn.LSTM(input_size=2*512, hidden_size=256, num_layers=2, batch_first=True).to(device)
    # 通过LSTM模型处理特征
    output, (hidden, cell) = lstm_model(features)

    # 使用最后一个隐藏状态作为音频的向量表示
    audio_representation = hidden[-1]
    
    x = audio_representation.detach().cpu().reshape(1, -1)[0]
    print("xxxxxxxx: ",x.size())
    return x #.numpy() #

# 进行聚类
def cluster_features(features, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return kmeans

# 保存分类结果
def save_clusters_to_csv(file_paths, labels, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'label'])
        for file_path, label in zip(file_paths, labels):
            print("csv write")
            writer.writerow([file_path, label])

# 绘图展示分类结果
def plot_clusters(labels):
    plt.hist(labels, bins=np.arange(-0.5, max(labels)+1.5, 1.0), rwidth=0.8)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Distribution of audio files in clusters')
    plt.savefig('/opt/my_figure.png')
    print("plot")
    # plt.show()

model_path = '/opt/chinese-wav2vec2-base-fairseq-ckpt.pt'
wav2vec_model = load_wav2vec_model(model_path)

audio_folder = '/opt/audio-s'
file_paths = [os.path.join(audio_folder, name) for name in os.listdir(audio_folder) if name.endswith(".mp3")]

features = [extract_features(wav2vec_model, file_path) for file_path in file_paths]
features = torch.stack(features, dim=0)
# print("features: ",np.array(features).shape)
kmeans = cluster_features(features, n_clusters=10)
labels = kmeans.labels_
print(labels)
save_clusters_to_csv(file_paths, labels, '/opt/clusters.csv')

plot_clusters(labels)
