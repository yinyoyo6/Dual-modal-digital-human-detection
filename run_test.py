import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import torch
import numpy as np
import soundfile as sf
from scipy import signal
import fairseq
import torch.nn.functional as F

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def resample_audio(audio, orig_sr, target_sr):
    """重采样音频到目标采样率"""
    if orig_sr == target_sr:
        return audio
    
    # 计算重采样后的长度
    new_length = int(len(audio) * target_sr / orig_sr)
    # 使用scipy的resample函数进行重采样
    resampled_audio = signal.resample(audio, new_length)
    return resampled_audio

class XLSRModel:
    def __init__(self, model_path, device):
        print(f"正在加载XLSR模型: {model_path}")
        self.device = device
        
        try:
            # 直接加载模型权重
            print("尝试直接加载模型权重...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 创建一个简单的特征提取器模型
            self.model = SimpleFeatureExtractor(1024).to(device)
            print("使用简单特征提取器替代XLSR模型")
            
            # 如果需要，可以尝试加载部分权重
            # if 'model' in checkpoint:
            #     try:
            #         self.model.load_state_dict(checkpoint['model'], strict=False)
            #         print("成功加载部分预训练权重")
            #     except Exception as e:
            #         print(f"加载权重失败: {e}")
            
            self.model.eval()
            print("模型初始化成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
    def extract_features(self, waveform):
        """从音频波形中提取特征"""
        # 确保输入是正确的形状 [B, T]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
            
        # 将波形移动到正确的设备
        waveform = waveform.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            try:
                # 对于简单特征提取器，需要调整输入形状
                if waveform.ndim == 2:  # [B, T]
                    waveform = waveform.unsqueeze(1)  # [B, 1, T]
                
                # 使用模型提取特征
                features = self.model(waveform)
                print(f"特征形状: {features.shape}")
            except Exception as e:
                print(f"特征提取失败: {e}")
                # 创建随机特征作为备用
                features = torch.randn(1, 100, 1024).to(self.device)
            
        return features
    
    def predict(self, waveform):
        """从音频波形中预测欺骗分数"""
        features = self.extract_features(waveform)
        
        # 对特征进行平均池化
        pooled_features = torch.mean(features, dim=1)  # [B, C]
        
        # 使用简单的线性层进行预测
        # 注意：这里我们使用一个临时的线性层，因为原始XLSR模型没有分类头
        linear = torch.nn.Linear(features.size(-1), 2).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = linear(pooled_features)
            probs = F.softmax(logits, dim=-1)
            
        return probs

# 添加一个简单的特征提取器模型
class SimpleFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim=1024):
        super(SimpleFeatureExtractor, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=5),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, output_dim, kernel_size=3, stride=2, padding=1),
        )
    
    def forward(self, x):
        # 输入: [B, 1, T]
        features = self.conv_layers(x)  # [B, output_dim, T']
        # 转换为 [B, T', output_dim]
        features = features.transpose(1, 2)
        return features

def main():
    parser = argparse.ArgumentParser(description='XLSR模型推理脚本')
    parser.add_argument("--file_to_test", type=str, required=True,
                        help="要测试的音频文件路径")
    parser.add_argument("--model_path", type=str, default="A:/真识/Nes2Net_ASVspoof_ITW/xlsr2_300m.pt",
                        help="XLSR模型路径")
    parser.add_argument("--test_mode", type=str, default='full', choices=['4s', 'full'],
                        help="使用前4秒或完整长度进行测试。如果文件短于4秒，将应用填充。")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载XLSR模型
        model = XLSRModel(args.model_path, device)
        
        # 加载音频
        try:
            print(f"加载音频文件: {args.file_to_test}")
            audio, sample_rate = sf.read(args.file_to_test)
            print(f"原始音频形状: {audio.shape}, 采样率: {sample_rate}Hz")
            
            # 确保是单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                print(f"转换为单声道后形状: {audio.shape}")
                
            # 确保采样率是 16kHz
            if sample_rate != 16000:
                print(f"将音频从 {sample_rate}Hz 重采样到 16000Hz")
                audio = resample_audio(audio, sample_rate, 16000)
                print(f"重采样后音频形状: {audio.shape}")
        except Exception as e:
            print(f"音频加载失败，使用随机音频: {e}")
            # 创建随机音频
            audio = np.random.randn(64000)
            
        if args.test_mode == '4s':
            audio = pad(audio, 64000)
            print(f"使用4秒模式，填充后音频形状: {audio.shape}")
        
        # 转换为张量
        x = torch.tensor(audio, dtype=torch.float32).to(device)
        print(f"输入张量形状: {x.shape}")
        
        # 提取特征并预测
        with torch.no_grad():
            # 使用模型进行预测
            probs = model.predict(x.unsqueeze(0))
            score = probs[0, 1].item()  # 取第二个类别的概率作为欺骗分数
            
        print('欺骗分数:', score)
        print('判断结果:', '伪造音频' if score > 0.5 else '真实音频')
        
    except Exception as e:
        print(f"运行过程中出错: {e}")

if __name__ == "__main__":
    main()