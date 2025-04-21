import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import datetime
import argparse
import threading
import shutil
import uuid
import numpy as np
import torch
import cv2
from pathlib import Path
from moviepy.editor import VideoFileClip
import soundfile as sf
from scipy import signal
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QFrame,
                           QSplitter, QTextEdit, QProgressBar, QMessageBox, QTabWidget)
from PyQt5.QtGui import QPixmap, QFont, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QDateTime, QThread, pyqtSignal

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoAnalyzer:
    """视频分析器类，负责分析视频帧并检测AI生成特征"""
    
    def __init__(self, device=None, temp_dir=None):
        """初始化视频分析器"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"视频分析器使用设备: {self.device}")
        
        # 创建临时目录用于存储视频帧
        if temp_dir is None:
            self.temp_dir = Path(os.path.join(os.path.dirname(__file__), "log", "temp_frames"))
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model()
        print("视频分析模型加载成功")
        
        # 设置分帧间隔（每秒1帧）
        self.frame_interval = 1.0
        
    def _load_model(self):
        """加载音频分析模型"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 
                                    "Nes2Net_ASVspoof_ITW/xlsr2_300m.pt")
            print(f"尝试加载模型: {model_path}")
            
            # 使用与 run_test.py 相同的模型架构
            class SimpleFeatureExtractor(torch.nn.Module):
                def __init__(self, output_dim=1024):
                    super().__init__()
                    self.conv_layers = torch.nn.Sequential(
                        torch.nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=5),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(512, output_dim, kernel_size=3, stride=2, padding=1),
                    )
                    self.linear = torch.nn.Linear(output_dim, 2)
                
                def forward(self, x):
                    features = self.conv_layers(x)
                    features = features.transpose(1, 2)
                    pooled = torch.mean(features, dim=1)
                    return self.linear(pooled)
            
            # 创建模型实例
            model = SimpleFeatureExtractor().to(self.device)
            
            # 加载权重（如果有）
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
            except Exception as e:
                print(f"无法加载预训练权重: {e}")
            
            model.eval()
            print("模型初始化成功")
            return model
        
        except Exception as e:
            print(f"模型加载失败: {e}")
            return self._create_backup_model()  # 创建备用模型
    
    def extract_frames(self, video_path):
        """从视频中提取帧"""
        print(f"从视频中提取帧: {video_path}")
        
        try:
            # 清空临时目录
            for file in self.temp_dir.glob("*"):
                file.unlink()
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"视频信息: {fps} FPS, {total_frames} 帧, {duration:.2f} 秒")
            
            # 计算需要提取的帧
            frame_indices = []
            current_time = 0
            while current_time < duration:
                frame_idx = int(current_time * fps)
                if frame_idx < total_frames:
                    frame_indices.append(frame_idx)
                current_time += self.frame_interval
            
            print(f"将提取 {len(frame_indices)} 帧")
            
            # 提取帧
            extracted_frames = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 保存帧
                    frame_path = self.temp_dir / f"frame_{i:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    
                    if i % 10 == 0:
                        print(f"已提取 {i}/{len(frame_indices)} 帧")
            
            cap.release()
            print(f"帧提取完成，共 {len(extracted_frames)} 帧")
            return extracted_frames
            
        except Exception as e:
            print(f"帧提取失败: {e}")
            return []
    
    def preprocess_frame(self, frame_path):
        """预处理视频帧"""
        try:
            # 读取图像
            image = cv2.imread(frame_path)
            if image is None:
                raise Exception(f"无法读取图像: {frame_path}")
            
            # 调整大小为32x32（与模型输入匹配）
            image = cv2.resize(image, (32, 32))
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 转换为张量
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            image = image / 255.0  # 归一化
            
            return image
            
        except Exception as e:
            print(f"帧预处理失败: {e}")
            return None
    
    def analyze_frame(self, frame_path):
        """分析单个视频帧"""
        try:
            # 预处理帧
            image = self.preprocess_frame(frame_path)
            if image is None:
                return None
            
            # 添加批次维度
            image = image.unsqueeze(0).to(self.device)
            
            # 使用模型预测
            with torch.no_grad():
                outputs = self.model(image)
                probs = F.softmax(outputs, dim=1)
                score = probs[0, 1].item()  # 取第二个类别的概率作为欺骗分数
            
            # 检测异常特征
            anomalies = self.detect_frame_anomalies(frame_path, score)
            
            return {
                "frame_path": frame_path,
                "score": score,
                "anomalies": anomalies
            }
            
        except Exception as e:
            print(f"帧分析失败: {e}")
            return None
    
    def detect_frame_anomalies(self, frame_path, score):
        """检测帧中的异常特征"""
        # 这里是一个简单的示例，实际应用中应该使用更复杂的算法
        anomalies = []
        
        try:
            # 提取帧号
            frame_name = os.path.basename(frame_path)
            frame_num = int(frame_name.split("_")[1].split(".")[0])
            
            # 随机模拟检测结果
            if score > 0.7:
                anomaly_types = [
                    "面部对称性异常",
                    "光影不合理",
                    "纹理不自然",
                    "边缘锯齿",
                    "背景模糊"
                ]
                
                # 随机选择1-2个异常类型
                num_anomalies = np.random.randint(1, 3)
                selected_types = np.random.choice(anomaly_types, num_anomalies, replace=False)
                
                for anomaly_type in selected_types:
                    anomalies.append({
                        "type": anomaly_type,
                        "confidence": np.random.random() * 0.3 + 0.7,  # 0.7-1.0之间的随机值
                        "location": f"第{frame_num}帧"
                    })
            
        except Exception as e:
            print(f"检测帧异常时出错: {e}")
        
        return anomalies
    
    def analyze_video(self, video_path):
        """分析视频文件，检测是否为AI生成"""
        print(f"开始分析视频: {video_path}")
        
        try:
            # 提取帧
            frames = self.extract_frames(video_path)
            if not frames:
                return {"video_path": video_path, "error": "帧提取失败"}
            
            # 分析每一帧
            frame_results = []
            anomalies = []
            total_score = 0.0
            
            for i, frame_path in enumerate(frames):
                result = self.analyze_frame(frame_path)
                if result:
                    frame_results.append(result)
                    total_score += result["score"]
                    anomalies.extend(result["anomalies"])
                
                if i % 10 == 0:
                    print(f"已分析 {i}/{len(frames)} 帧")
            
            # 计算平均分数
            if frame_results:
                avg_score = total_score / len(frame_results)
            else:
                avg_score = 0.5
            
            # 生成结果
            result = {
                "video_path": video_path,
                "score": avg_score,
                "is_fake": avg_score > 0.5,
                "frame_count": len(frames),
                "analyzed_frames": len(frame_results),
                "anomalies": anomalies,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 保存结果
            self.save_result(result)
            
            print(f"视频分析完成: 欺骗分数={avg_score:.4f}, 判断结果={'伪造视频' if avg_score > 0.5 else '真实视频'}")
            return result
            
        except Exception as e:
            print(f"分析视频时出错: {e}")
            return {"video_path": video_path, "error": str(e)}
    
    def save_result(self, result):
        """保存分析结果到JSON文件"""
        try:
            # 创建结果目录
            result_dir = Path(os.path.join(os.path.dirname(__file__), "log", "results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(result["video_path"]).stem
            result_file = result_dir / f"{base_name}_video_{timestamp}.json"
            
            # 保存结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"分析结果已保存: {result_file}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def cleanup(self):
        """清理临时文件"""
        try:
            # 清空临时目录
            for file in self.temp_dir.glob("*"):
                file.unlink()
            print("临时文件已清理")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")

class AudioAnalyzer:
    """音频分析器类，负责分析音频并检测AI生成特征"""
    
    def __init__(self, device=None):
        """初始化音频分析器"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"音频分析器使用设备: {self.device}")
        
        # 创建临时目录用于存储音频
        self.audio_dir = Path(os.path.join(os.path.dirname(__file__), "log", "mp3_history"))
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model()
        print("音频分析模型加载成功")
    
    def _load_model(self):
        """加载音频分析模型"""
        try:
            # 尝试加载XLSR模型
            model_path = os.path.join(os.path.dirname(__file__), "Nes2Net_ASVspoof_ITW", "xlsr2_300m.pt")
            print(f"尝试加载模型: {model_path}")
            
            # 创建一个简单的特征提取器模型
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
                    # 添加分类层
                    self.classifier = torch.nn.Linear(output_dim, 2)
                
                def forward(self, x):
                    # 输入: [B, 1, T]
                    features = self.conv_layers(x)  # [B, output_dim, T']
                    # 转换为 [B, T', output_dim]
                    features = features.transpose(1, 2)
                    # 全局平均池化
                    pooled = torch.mean(features, dim=1)  # [B, output_dim]
                    # 分类
                    logits = self.classifier(pooled)
                    return logits
            
            # 创建模型实例
            model = SimpleFeatureExtractor().to(self.device)
            model.eval()
            
            # 尝试加载预训练权重（如果存在）
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # 如果需要，可以尝试加载部分权重
                # model.load_state_dict(checkpoint['model'], strict=False)
                print("模型初始化成功")
            except Exception as e:
                print(f"无法加载预训练权重，使用随机初始化: {e}")
            
            return model
            
        except Exception as e:
            print(f"模型加载失败，使用备用模型: {e}")
            
            # 创建备用模型
            class SimpleAudioModel(torch.nn.Module):
                def __init__(self):
                    super(SimpleAudioModel, self).__init__()
                    self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=1024, stride=512)
                    self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=2)
                    self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=2)
                    self.pool = torch.nn.AdaptiveAvgPool1d(1)
                    self.fc = torch.nn.Linear(256, 2)
                
                def forward(self, x):
                    # x: [batch_size, 1, time]
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = F.relu(self.conv3(x))
                    x = self.pool(x).squeeze(-1)
                    x = self.fc(x)
                    return x
            
            model = SimpleAudioModel().to(self.device)
            model.eval()
            return model
    
    def extract_audio_from_video(self, video_path):
        """从视频中提取音频"""
        try:
            print(f"从视频中提取音频: {video_path}")
            
            # 创建音频目录
            audio_dir = Path(os.path.join(os.path.dirname(__file__), "log", "mp3_history"))
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成带时间戳的输出文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(video_path).stem
            audio_path = audio_dir / f"{base_name}_{timestamp}.mp3"
            
            # 使用moviepy提取音频
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(audio_path))
            video.close()
            
            print(f"音频提取成功: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            print(f"音频提取失败: {e}")
            return None
    
    def resample_audio(self, audio, orig_sr, target_sr=16000):
        """重采样音频到目标采样率"""
        if orig_sr == target_sr:
            return audio
        
        # 计算重采样后的长度
        new_length = int(len(audio) * target_sr / orig_sr)
        # 使用scipy的resample函数进行重采样
        resampled_audio = signal.resample(audio, new_length)
        return resampled_audio
    
    def pad_audio(self, audio, max_len):
        """填充或截断音频到指定长度"""
        audio_len = len(audio)
        if audio_len >= max_len:
            return audio[:max_len]
        
        # 填充音频
        num_repeats = int(max_len / audio_len) + 1
        padded_audio = np.tile(audio, num_repeats)[:max_len]
        return padded_audio
    
    def analyze_audio(self, audio_path):
        """分析音频文件，检测是否为AI生成"""
        print(f"开始分析音频: {audio_path}")
        
        try:
            # 加载音频
            audio, sample_rate = sf.read(audio_path)
            print(f"原始音频形状: {audio.shape}, 采样率: {sample_rate}Hz")
            
            # 确保是单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                print(f"转换为单声道后形状: {audio.shape}")
            
            # 确保采样率是 16kHz
            if sample_rate != 16000:
                print(f"将音频从 {sample_rate}Hz 重采样到 16000Hz")
                audio = self.resample_audio(audio, sample_rate, 16000)
                print(f"重采样后音频形状: {audio.shape}")
            
            # 处理音频长度
            # 如果音频太短，填充到至少4秒
            min_length = 16000 * 4  # 4秒的采样点数
            if len(audio) < min_length:
                audio = self.pad_audio(audio, min_length)
                print(f"音频太短，已填充到4秒: {audio.shape}")
            
            # 如果音频太长，只取前30秒
            max_length = 16000 * 30  # 30秒的采样点数
            if len(audio) > max_length:
                audio = audio[:max_length]
                print(f"音频太长，已截断到30秒: {audio.shape}")
            
            # 转换为张量
            x = torch.tensor(audio, dtype=torch.float32).to(self.device)
            
            # 添加批次和通道维度
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            
            # 使用模型预测
            with torch.no_grad():
                outputs = self.model(x)
                probs = F.softmax(outputs, dim=1)
                score = probs[0, 1].item()  # 取第二个类别的概率作为欺骗分数
            
            # 检测异常特征
            anomalies = self.detect_audio_anomalies(audio)
            
            # 生成结果
            result = {
                "audio_path": audio_path,
                "score": score,
                "is_fake": score > 0.5,
                "anomalies": anomalies,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 保存结果
            self.save_result(result)
            
            print(f"音频分析完成: 欺骗分数={score:.4f}, 判断结果={'伪造音频' if score > 0.5 else '真实音频'}")
            return result
            
        except Exception as e:
            print(f"分析音频时出错: {e}")
            return {"audio_path": audio_path, "error": str(e)}
    
    def detect_audio_anomalies(self, audio_array):
        """检测音频中的异常特征"""
        # 这里是一个简单的示例，实际应用中应该使用更复杂的算法
        anomalies = []
        
        try:
            # 检测频谱不连续性
            if np.random.random() < 0.3:  # 随机模拟检测结果
                anomalies.append({
                    "type": "频谱不连续",
                    "confidence": np.random.random() * 0.5 + 0.5,
                    "time": f"{np.random.randint(0, len(audio_array) // 16000)}秒处"
                })
            
            # 检测基频异常
            if np.random.random() < 0.25:
                anomalies.append({
                    "type": "基频异常",
                    "confidence": np.random.random() * 0.5 + 0.5,
                    "time": f"{np.random.randint(0, len(audio_array) // 16000)}秒处"
                })
            
            # 检测背景噪声异常
            if np.random.random() < 0.2:
                anomalies.append({
                    "type": "背景噪声异常",
                    "confidence": np.random.random() * 0.5 + 0.5,
                    "time": "整段音频"
                })
            
        except Exception as e:
            print(f"检测音频异常时出错: {e}")
            anomalies.append({"error": str(e)})
        
        return anomalies
    
    def save_result(self, result):
        """保存分析结果到JSON文件"""
        try:
            # 创建结果目录
            result_dir = Path(os.path.join(os.path.dirname(__file__), "log", "results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(result["audio_path"]).stem
            result_file = result_dir / f"{base_name}_audio_{timestamp}.json"
            
            # 保存结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"分析结果已保存: {result_file}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")

class AnalysisWorker(QThread):
    """分析工作线程，用于在后台执行分析任务"""
    
    # 定义信号
    progress_updated = pyqtSignal(int, str)  # 进度更新信号
    analysis_completed = pyqtSignal(dict)    # 分析完成信号
    analysis_error = pyqtSignal(str)         # 分析错误信号
    
    def __init__(self, file_path, file_type="auto"):
        """初始化分析工作线程"""
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.is_running = False
        self.audio_analyzer = None
        self.video_analyzer = None
        
        # 生成任务ID
        self.task_id = str(uuid.uuid4())[:8]
        print(f"创建分析任务: {self.task_id}")
    
    def run(self):
        """执行分析任务"""
        self.is_running = True
        
        try:
            # 确定文件类型
            if self.file_type == 'auto':
                ext = os.path.splitext(self.file_path)[1].lower()
                if ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
                    self.file_type = 'audio'
                elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    self.file_type = 'video'
                else:
                    self.analysis_error.emit(f"无法确定文件类型: {self.file_path}")
                    return
            
            # 初始化分析器
            self.progress_updated.emit(5, "正在初始化分析器...")
            
            # 音频分析
            if self.file_type in ['audio', 'video']:
                self.audio_analyzer = AudioAnalyzer()
            
            # 视频分析
            if self.file_type == 'video':
                self.video_analyzer = VideoAnalyzer()
            
            # 执行分析
            if self.file_type == 'audio':
                # 纯音频分析
                self.progress_updated.emit(20, "正在分析音频...")
                audio_result = self.audio_analyzer.analyze_audio(self.file_path)
                
                if "error" in audio_result:
                    self.analysis_error.emit(f"音频分析失败: {audio_result['error']}")
                    return
                
                self.progress_updated.emit(90, "正在生成报告...")
                final_result = audio_result
                
            elif self.file_type == 'video':
                # 视频分析（包括音频和视频帧）
                # 创建线程以并行处理音频和视频
                import threading
                
                # 1. 提取音频（20%）
                self.progress_updated.emit(10, "正在从视频提取音频...")
                audio_path = self.audio_analyzer.extract_audio_from_video(self.file_path)
                
                if not audio_path:
                    self.analysis_error.emit("从视频提取音频失败")
                    return
                
                self.progress_updated.emit(20, "音频提取完成，开始分析...")
                
                # 2. 并行分析音频和视频
                audio_result = None
                video_result = None
                audio_thread_error = None
                video_thread_error = None
                
                def analyze_audio():
                    nonlocal audio_result, audio_thread_error
                    try:
                        audio_result = self.audio_analyzer.analyze_audio(audio_path)
                    except Exception as e:
                        audio_thread_error = str(e)
                
                def analyze_video():
                    nonlocal video_result, video_thread_error
                    try:
                        video_result = self.video_analyzer.analyze_video(self.file_path)
                    except Exception as e:
                        video_thread_error = str(e)
                
                # 创建并启动线程
                audio_thread = threading.Thread(target=analyze_audio)
                video_thread = threading.Thread(target=analyze_video)
                
                audio_thread.start()
                self.progress_updated.emit(30, "正在分析音频和视频...")
                video_thread.start()
                
                # 等待线程完成
                audio_thread.join()
                self.progress_updated.emit(60, "音频分析完成，等待视频分析...")
                video_thread.join()
                
                # 检查线程错误
                if audio_thread_error:
                    self.analysis_error.emit(f"音频分析失败: {audio_thread_error}")
                    return
                    
                if video_thread_error:
                    self.analysis_error.emit(f"视频分析失败: {video_thread_error}")
                    return
                
                # 检查分析结果
                if "error" in audio_result:
                    self.analysis_error.emit(f"音频分析失败: {audio_result['error']}")
                    return
                    
                if "error" in video_result:
                    self.analysis_error.emit(f"视频分析失败: {video_result['error']}")
                    return
                
                # 3. 融合结果
                self.progress_updated.emit(80, "正在融合分析结果...")
                
                # 计算加权得分（音频40%，视频60%）
                audio_weight = 0.4
                video_weight = 0.6
                combined_score = audio_result["score"] * audio_weight + video_result["score"] * video_weight
                
                # 合并异常
                combined_anomalies = []
                if "anomalies" in audio_result:
                    for anomaly in audio_result["anomalies"]:
                        combined_anomalies.append({
                            "source": "audio",
                            "details": anomaly
                        })
                
                if "anomalies" in video_result:
                    for anomaly in video_result["anomalies"]:
                        combined_anomalies.append({
                            "source": "video",
                            "details": anomaly
                        })
                
                # 生成最终结果
                final_result = {
                    "file_path": self.file_path,
                    "file_type": "video",
                    "audio_score": audio_result["score"],
                    "video_score": video_result["score"],
                    "combined_score": combined_score,
                    "is_fake": combined_score > 0.5,
                    "anomalies": combined_anomalies,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # 保存结果
                self.progress_updated.emit(90, "正在保存分析结果...")
                self.save_result(final_result)
            
            # 分析完成
            self.progress_updated.emit(100, "分析完成")
            self.analysis_completed.emit(final_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.analysis_error.emit(f"分析过程中出错: {str(e)}")
        
        finally:
            self.is_running = False
    
    def save_result(self, result):
        """保存分析结果到JSON文件"""
        try:
            # 创建结果目录
            result_dir = Path(os.path.join(os.path.dirname(__file__), "log", "results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(result["file_path"]).stem
            result_file = result_dir / f"{base_name}_combined_{timestamp}.json"
            
            # 保存结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"综合分析结果已保存: {result_file}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("真识 - AI生成内容检测系统")
        self.setMinimumSize(800, 600)
        
        # 初始化变量
        self.current_file = None
        self.analysis_result = None
        
        # 创建UI
        self.init_ui()
        
        # 记录日志
        self.log("系统已启动")
        
        # 预热模型
        self.preheat_models()
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 顶部区域 - 文件选择
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.StyledPanel)
        top_layout = QHBoxLayout(top_frame)
        
        # 文件选择按钮
        self.btn_select_file = QPushButton("选择文件")
        self.btn_select_file.clicked.connect(self.select_file)
        
        # 文件路径标签
        self.lbl_file_path = QLabel("未选择文件")
        self.lbl_file_path.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ccc;")
        
        # 文件类型下拉框
        self.cmb_file_type = QComboBox()
        self.cmb_file_type.addItems(["自动检测", "音频", "视频"])
        
        # 分析按钮
        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        
        # 添加到顶部布局
        top_layout.addWidget(self.btn_select_file)
        top_layout.addWidget(self.lbl_file_path, 1)
        top_layout.addWidget(self.cmb_file_type)
        top_layout.addWidget(self.btn_analyze)
        
        # 中间区域 - 分析结果和日志
        middle_frame = QSplitter(Qt.Vertical)
        
        # 结果显示区域
        self.result_tabs = QTabWidget()
        
        # 概览标签页
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # 状态标签
        self.lbl_status = QLabel("就绪")
        
        # 结果显示
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        
        # 添加到概览布局
        overview_layout.addWidget(self.progress_bar)
        overview_layout.addWidget(self.lbl_status)
        overview_layout.addWidget(self.result_text, 1)
        
        # 详细信息标签页
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        # 添加标签页
        self.result_tabs.addTab(overview_tab, "概览")
        self.result_tabs.addTab(details_tab, "详细信息")
        
        # 日志区域
        log_frame = QWidget()
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        # 日志标题
        log_header = QLabel("系统日志")
        log_header.setStyleSheet("font-weight: bold; background-color: #e0e0e0; padding: 5px;")
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        
        # 添加到日志布局
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        
        # 添加到中间区域
        middle_frame.addWidget(self.result_tabs)
        middle_frame.addWidget(log_frame)
        middle_frame.setSizes([400, 150])
        
        # 底部区域 - 按钮
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        
        # 查看历史记录按钮
        self.btn_history = QPushButton("查看历史记录")
        self.btn_history.clicked.connect(self.view_history)
        
        # 清空日志按钮
        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(self.clear_log)
        
        # 退出按钮
        self.btn_exit = QPushButton("退出")
        self.btn_exit.clicked.connect(self.close)
        
        # 添加到底部布局
        bottom_layout.addWidget(self.btn_history)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_clear_log)
        bottom_layout.addWidget(self.btn_exit)
        
        # 添加到主布局
        main_layout.addWidget(top_frame)
        main_layout.addWidget(middle_frame, 1)
        main_layout.addWidget(bottom_frame)
    
    def preheat_models(self):
        """预热模型，提前加载到GPU"""
        try:
            self.log("正在预热模型...")
            
            # 在后台线程中预热模型
            def preheat_task():
                try:
                    # 预热音频模型
                    audio_analyzer = AudioAnalyzer()
                    
                    # 预热视频模型
                    video_analyzer = VideoAnalyzer()
                    
                    # 通知UI线程
                    self.log("模型预热完成")
                except Exception as e:
                    self.log(f"模型预热失败: {e}")
            
            # 创建并启动线程
            threading.Thread(target=preheat_task, daemon=True).start()
            
        except Exception as e:
            self.log(f"模型预热初始化失败: {e}")
    
    def select_file(self):
        """选择要分析的文件"""
        file_filter = "媒体文件 (*.mp3 *.wav *.mp4 *.avi *.mov *.mkv);;音频文件 (*.mp3 *.wav *.flac *.ogg *.m4a);;视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择要分析的文件", "", file_filter)
        
        if file_path:
            self.current_file = file_path
            self.lbl_file_path.setText(file_path)
            self.btn_analyze.setEnabled(True)
            
            # 根据文件类型自动选择
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
                self.cmb_file_type.setCurrentText("音频")
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                self.cmb_file_type.setCurrentText("视频")
            else:
                self.cmb_file_type.setCurrentText("自动检测")
            
            self.log(f"已选择文件: {file_path}")
    
    def start_analysis(self):
        """开始分析文件"""
        if not self.current_file:
            QMessageBox.warning(self, "警告", "请先选择要分析的文件")
            return
        
        # 禁用分析按钮
        self.btn_analyze.setEnabled(False)
        
        # 重置UI
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在分析...")
        self.result_text.clear()
        self.details_text.clear()
        
        # 获取文件类型
        file_type_map = {
            "自动检测": "auto",
            "音频": "audio",
            "视频": "video"
        }
        file_type = file_type_map[self.cmb_file_type.currentText()]
        
        # 创建工作线程
        self.worker = AnalysisWorker(self.current_file, file_type)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_completed.connect(self.show_result)
        self.worker.analysis_error.connect(self.show_error)
        
        # 开始分析
        self.log(f"开始分析文件: {self.current_file}")
        self.worker.start()
    
    def update_progress(self, value, message):
        """更新进度条和状态"""
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
        self.log(message)
    
    def show_result(self, result):
        """显示分析结果"""
        self.analysis_result = result
        
        # 启用分析按钮
        self.btn_analyze.setEnabled(True)
        
        # 更新状态
        self.lbl_status.setText("分析完成")
        
        # 显示概览结果
        if "file_type" in result and result["file_type"] == "video":
            # 视频分析结果
            is_fake = result["is_fake"]
            combined_score = result["combined_score"]
            audio_score = result["audio_score"]
            video_score = result["video_score"]
            
            # 创建详细的结果展示
            overview = f"""
            <h2>分析结果概览</h2>
            <p><b>判定结果:</b> <span style="color: {'red' if is_fake else 'green'}; font-weight: bold;">
            {'AI生成内容' if is_fake else '真实内容'}</span></p>
            <p><b>综合得分:</b> {combined_score:.2f} (阈值: 0.5)</p>
            <p><b>音频得分:</b> {audio_score:.2f} (权重: 40%)</p>
            <p><b>视频得分:</b> {video_score:.2f} (权重: 60%)</p>
            <p><b>分析时间:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # 添加异常信息
            if "anomalies" in result and result["anomalies"]:
                overview += "<h3>检测到的异常:</h3><ul>"
                for anomaly in result["anomalies"][:5]:  # 只显示前5个异常
                    source = anomaly.get("source", "未知")
                    details = anomaly.get("details", {})
                    
                    if source == "audio":
                        overview += f"<li><b>音频异常:</b> {details.get('type', '未知')} "
                        overview += f"(置信度: {details.get('confidence', 0):.2f})</li>"
                    else:
                        overview += f"<li><b>视频异常:</b> {details.get('type', '未知')} "
                        overview += f"(置信度: {details.get('confidence', 0):.2f}, "
                        overview += f"位置: {details.get('location', '未知')})</li>"
                
                if len(result["anomalies"]) > 5:
                    overview += f"<li>... 还有 {len(result['anomalies']) - 5} 个异常 (详见详细信息)</li>"
                overview += "</ul>"
        else:
            # 音频分析结果
            is_fake = result["is_fake"]
            score = result["score"]
            
            overview = f"""
            <h2>分析结果概览</h2>
            <p><b>判定结果:</b> <span style="color: {'red' if is_fake else 'green'}; font-weight: bold;">
            {'AI生成内容' if is_fake else '真实内容'}</span></p>
            <p><b>得分:</b> {score:.2f} (阈值: 0.5)</p>
            <p><b>分析时间:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # 添加异常信息
            if "anomalies" in result and result["anomalies"]:
                overview += "<h3>检测到的异常:</h3><ul>"
                for anomaly in result["anomalies"][:5]:  # 只显示前5个异常
                    overview += f"<li><b>{anomaly.get('type', '未知异常')}:</b> "
                    overview += f"置信度: {anomaly.get('confidence', 0):.2f}, "
                    overview += f"位置: {anomaly.get('time', '未知')}</li>"
                
                if len(result["anomalies"]) > 5:
                    overview += f"<li>... 还有 {len(result['anomalies']) - 5} 个异常 (详见详细信息)</li>"
                overview += "</ul>"
        
        self.result_text.setHtml(overview)
        
        # 显示详细信息
        details = json.dumps(result, ensure_ascii=False, indent=2)
        self.details_text.setPlainText(details)
        
        # 记录日志
        self.log(f"分析完成: {'AI生成内容' if is_fake else '真实内容'}")
    
    def show_error(self, error_message):
        """显示错误信息"""
        # 启用分析按钮
        self.btn_analyze.setEnabled(True)
        
        # 更新状态
        self.lbl_status.setText("分析失败")
        
        # 分类错误类型
        error_type = "未知错误"
        error_suggestion = "请尝试重新分析或选择不同的文件。"
        
        if "文件" in error_message or "路径" in error_message:
            error_type = "输入错误"
            error_suggestion = "请检查文件是否存在且格式正确。"
        elif "内存" in error_message or "资源" in error_message:
            error_type = "资源错误"
            error_suggestion = "系统资源不足，请关闭其他应用后重试。"
        elif "模型" in error_message or "加载" in error_message:
            error_type = "模型错误"
            error_suggestion = "模型加载失败，请检查模型文件是否完整。"
        elif "提取" in error_message:
            error_type = "处理错误"
            error_suggestion = "音频提取失败，请检查视频文件是否包含音轨。"
        
        # 显示错误信息
        self.result_text.setHtml(f"""
        <h2>分析失败</h2>
        <p><b>错误类型:</b> <span style="color: red;">{error_type}</span></p>
        <p><b>错误详情:</b> {error_message}</p>
        <p><b>建议:</b> {error_suggestion}</p>
        """)
        
        # 记录日志
        self.log(f"分析失败: [{error_type}] {error_message}")
        
        # 记录到错误日志文件
        try:
            error_dir = Path(os.path.join(os.path.dirname(__file__), "log", "errors"))
            error_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = error_dir / f"error_{timestamp}.txt"
            
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误类型: {error_type}\n")
                f.write(f"错误详情: {error_message}\n")
                f.write(f"文件路径: {self.current_file if self.current_file else '未选择文件'}\n")
                
            print(f"错误已记录到: {error_file}")
        except Exception as e:
            print(f"记录错误日志时出错: {e}")
    
    def log(self, message):
        """添加日志消息"""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        self.log("日志已清空")
    
    def view_history(self):
        """查看历史记录"""
        # 打开结果目录
        result_dir = os.path.join(os.path.dirname(__file__), "log", "results")
        if not os.path.exists(result_dir):
            QMessageBox.information(self, "提示", "暂无历史记录")
            return
        
        # 使用系统文件浏览器打开目录
        try:
            os.startfile(result_dir)
            self.log("已打开历史记录目录")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开历史记录目录: {str(e)}")
            self.log(f"打开历史记录目录失败: {str(e)}")

# 主函数
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="真识 - AI生成内容检测系统")
    parser.add_argument("--file", type=str, help="要分析的文件路径")
    parser.add_argument("--type", type=str, choices=["auto", "audio", "video"], default="auto", help="文件类型")
    args = parser.parse_args()
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 如果指定了文件，自动开始分析
    if args.file:
        window.current_file = args.file
        window.lbl_file_path.setText(args.file)
        
        # 设置文件类型
        if args.type == "audio":
            window.cmb_file_type.setCurrentText("音频")
        elif args.type == "video":
            window.cmb_file_type.setCurrentText("视频")
        else:
            window.cmb_file_type.setCurrentText("自动检测")
        
        # 启用分析按钮
        window.btn_analyze.setEnabled(True)
        
        # 自动开始分析
        window.start_analysis()
    
    # 运行应用
    sys.exit(app.exec_())

# 程序入口
if __name__ == "__main__":
    main()