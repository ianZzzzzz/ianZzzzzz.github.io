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
