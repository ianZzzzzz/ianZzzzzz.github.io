# 我希望能够用wav2vec的输出，自监督的训练一个lstm用作语音向量的编码器,请给我一份完整的训练代码，包含tensorboard

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from fairseq.models.wav2vec import Wav2VecModel
from torch.optim import Adam
from fairseq import checkpoint_utils
device = torch.device("cuda:0")
# 使用TensorBoard记录训练过程
# 创建SummaryWriter，指定日志目录
log_dir = "logs"
writer = SummaryWriter(log_dir)

class AudioDataset(Dataset):
    def __init__(self, directory, device):
        self.directory = directory
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]
        self.device = device

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # 将左右声道合并
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 将数据移动到GPU并转换为HalfTensor
        waveform = waveform.to(self.device)#.half()

        return waveform.squeeze(0)
    

def load_wav2vec(model_path):
    print("loading model(s) from {}".format(model_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path],suffix="",)
    model = models[0]
    model = model#.half()
    wav2vec_model = model.to(device)
    wav2vec_model.eval()
    return wav2vec_model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 新的线性层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 应用线性层

        return out

# 加载wav2vec模型
wav2vec_model = load_wav2vec('/opt/chinese-wav2vec2-base-fairseq-ckpt.pt')
# 定义数据集
audio_dataset = AudioDataset('/opt/audio-s',device)
# 定义数据加载器
audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=True)
# 定义LSTM模型
# lstm_model = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True).half() 
lstm_model = LSTM(input_size=512, hidden_size=256, num_layers=2, num_classes=512).to(device)#.half()
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 20


wav2vec_model = wav2vec_model.to(device)
# lstm_model = lstm_model.to(device)

for epoch in range(num_epochs):
    for i, audio_batch in enumerate(audio_loader):
        # 将数据移动到正确的设备上，并将其转换为正确的数据类型
        audio_batch = audio_batch.to(device)#.half()  # 如果你的模型是半精度的
        # 提取特征

        wav2vec_features = wav2vec_model.feature_extractor(audio_batch)
        wav2vec_features = wav2vec_features.transpose(1,2)
        # 使用LSTM模型处理特征
        
        output = lstm_model(wav2vec_features[:,:-1,:])
        # 定义目标
        targets = wav2vec_features[:,-1,:]

        # 计算损失
        loss = criterion(output, targets)  #output[:, :-1]

        # 计算梯度并更新权重
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录损失
        writer.add_scalar('Loss/train', loss, epoch)
        print('Loss/train: ', loss, " epoch:",epoch)

# 保存模型
torch.save(lstm_model.state_dict(), 'lstm_model.pt')
