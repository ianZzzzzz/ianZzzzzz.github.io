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
writer = SummaryWriter()
class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # 将左右声道合并
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # 将数据移动到GPU并转换为HalfTensor
        waveform = waveform.to(device).half()
         if self.transform:
             waveform = self.transform(waveform)

        return waveform.squeeze(0) 

def load_wav2vec(model_path):
    print("loading model(s) from {}".format(model_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path],suffix="",)
    model = models[0]
    model = model.half()
    wav2vec_model = model.to(device)
    wav2vec_model.eval()
    return wav2vec_model

# 加载wav2vec模型
wav2vec_model = load_wav2vec('/opt/chinese-wav2vec2-base-fairseq-ckpt.pt')
# 定义数据集
audio_dataset = AudioDataset('/opt/audio-s', transform=wav2vec_model.feature_extractor)
# 定义数据加载器
audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=True)
# 定义LSTM模型
lstm_model = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 20


wav2vec_model = wav2vec_model.to(device)
lstm_model = lstm_model.to(device)
for epoch in range(num_epochs):
    for i, audio_batch in enumerate(audio_loader):
        # 将数据移动到正确的设备上，并将其转换为正确的数据类型
        audio_batch = audio_batch.to(device).half()  # 如果你的模型是半精度的
        # 提取特征
        wav2vec_features = wav2vec_model.feature_extractor(audio_batch)

        # 使用LSTM模型处理特征
        output, (hidden, cell) = lstm_model(wav2vec_features)

        # 定义目标
        targets = wav2vec_features[:, 1:]

        # 计算损失
        loss = criterion(output[:, :-1], targets)

        # 计算梯度并更新权重
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录损失
        writer.add_scalar('Loss/train', loss, epoch)

# 保存模型
torch.save(lstm_model.state_dict(), 'lstm_model.pt')
