import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QSize, QTimer
import cv2
from models.vgg import VGG
from models.densenet import DenseNet121
from models.googlenet import GoogLeNet
from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet50
import numpy as np
import torch
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms,models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classes = os.listdir('./data')
classes.sort()
#------------------------------------------------------1.加载模型--------------------------------------------------------------
num_classes = len(classes)

try:
    path_model = "./logs/model.ckpt"
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"模型文件 {path_model} 不存在")
    model = torch.load(path_model, map_location=device)
    model = model.to(device)
    model.eval()  # 设置为评估模式
except Exception as e:
    print(f"模型加载错误: {str(e)}")
    sys.exit(1)

# 根据图片文件路径获取图像数据矩阵
def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image

# 模型预测前必要的图像处理
def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_chw = preprocess(input_image)
    return img_chw  # chw:channel height width

class ImageRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        # 添加结果缓存和计数器
        self.frame_results = []  # 存储最近n帧的结果
        self.frame_window = 30   # 滑动窗口大小(1秒)
        self.result_threshold = 0.7  # 置信度阈值
        self.init_ui()

    def init_ui(self):
        # Set up the main layout
        main_layout = QHBoxLayout(self)

        # Left side for loading images
        left_layout = QVBoxLayout()
        self.image_label = QLabel("Click here to load an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 200)
        self.image_label.setStyleSheet("border: 2px solid black")
        self.image_label.mousePressEvent = self.load_file
        left_layout.addWidget(self.image_label)

        # 添加视频控制按钮
        video_controls = QHBoxLayout()
        self.play_pause_btn = QPushButton("暂停")
        self.play_pause_btn.clicked.connect(self.toggle_video)
        self.play_pause_btn.setVisible(False)
        video_controls.addWidget(self.play_pause_btn)
    
        left_layout.addLayout(video_controls)

        # Middle side for recognition
        middle_layout = QVBoxLayout()
        recognize_button = QPushButton("Recognize")
        recognize_button.setStyleSheet("border: 2px solid black")
        recognize_button.clicked.connect(self.recognize_image)
        recognize_button.setMinimumSize(QSize(100, 50))
        middle_layout.addStretch(1)
        middle_layout.addWidget(recognize_button, alignment=Qt.AlignCenter)
        middle_layout.addStretch(1)

        # Right side for displaying results
        right_layout = QVBoxLayout()
        self.result_label = QLabel("Recognition result will be displayed here")
        self.result_label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 14, QFont.Bold)
        self.result_label.setFont(font)
        self.result_label.setStyleSheet("border: 2px solid black")
        right_layout.addWidget(self.result_label)

        # Add the layouts to the main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        self.setGeometry(100, 100, 800, 400)
        self.setWindowTitle("Image Recognition App")

        self.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #ffcccc, stop:1 #99ccff);")

    def load_file(self, event):
        options = QFileDialog.Options()
        file_filter = "Media Files (*.png *.jpg *.bmp *.gif *.mp4 *.avi *.mov);;Images (*.png *.jpg *.bmp *.gif);;Videos (*.mp4 *.avi *.mov)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter, options=options)
        
        if file_name:
            # 如果之前有视频在播放，先停止并清理资源
            if hasattr(self, 'cap') and self.cap is not None:
                self.timer.stop()
                self.cap.release()
                self.cap = None
                self.video_playing = False
                self.play_pause_btn.setVisible(False)
                self.frame_results.clear()  # 清空之前的分析结果
            
            self.file_name = file_name
            # 检查是否为视频文件
            self.is_video = self.file_name.lower().endswith(('.mp4', '.avi', '.mov'))
            
            if self.is_video:
                self.cap = cv2.VideoCapture(self.file_name)
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(30)  # 30ms 更新一次，约 33fps
                self.video_playing = True
                self.play_pause_btn.setVisible(True)
                self.play_pause_btn.setText("暂停")
            else:
                # 图片处理
                pixmap = QPixmap(self.file_name)
                pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                self.result_label.setText("Recognition result will be displayed here")
                
            self.image_label.setAlignment(Qt.AlignCenter)

    def update_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # 处理当前帧
            analysis, current_score = self.process_video_frame(frame)
            
            # 更新显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (300, 200))
            h, w, ch = frame_resized.shape
            qt_image = QImage(frame_resized.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            
            # 更新结果显示
            if analysis:
                self.result_label.setText(analysis)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def recognize_image(self):
        try:
            if not hasattr(self, 'file_name') or not self.file_name:
                self.result_label.setText("请先选择文件")
                return
                
            if hasattr(self, 'is_video') and self.is_video:
                # 视频处理：获取当前帧
                ret, frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_image = Image.fromarray(frame_rgb)
                else:
                    self.result_label.setText("无法读取视频帧")
                    return
            else:
                # 图片处理
                input_image = get_imageNdarray(self.file_name)
                
            # 后续处理保持不变
            input_image = input_image.resize((32,32))
            img_chw = process_imageNdarray(input_image)
            img_chw = img_chw.view(1, 3, 32, 32).to(device)
            
            with torch.no_grad():
                out = model(img_chw)
                probabilities = torch.nn.functional.softmax(out, dim=1)[0]
                score = probabilities.max().item() * 100
                predicted_idx = probabilities.argmax().item()
                result = classes[predicted_idx]
                
                result_text = f"预测结果: {result}\n置信度: {score:.2f}%"
                self.result_label.setText(result_text)
                
        except Exception as e:
            self.result_label.setText(f"识别出错: {str(e)}")

    def process_video_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(frame_rgb)
            input_image = input_image.resize((32, 32))
            img_chw = process_imageNdarray(input_image)
            img_chw = img_chw.view(1, 3, 32, 32).to(device)
            
            with torch.no_grad():
                out = model(img_chw)
                probabilities = torch.nn.functional.softmax(out, dim=1)[0]
                score = probabilities.max().item()
                predicted_idx = probabilities.argmax().item()
                result = classes[predicted_idx].lower()  # 转换为小写
                
                # 添加调试信息
                print(f"Frame analysis - Result: {result}, Score: {score:.3f}")
                
                if score > self.result_threshold:
                    self.frame_results.append((result, score))
                
                if len(self.frame_results) > self.frame_window:
                    self.frame_results.pop(0)
                
                analysis = self.analyze_frame_results()
                return analysis, score
                
        except Exception as e:
            print(f"帧处理错误: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈
            return None, 0

    def analyze_frame_results(self):
        if not self.frame_results:
            return "等待分析..."
        
        # 修改为小写处理
        votes = {'real': 0, 'fake': 0}
        for result, score in self.frame_results:
            # 将结果转换为小写
            result = result.lower()
            if result in votes:
                votes[result] += score
        
        # 防止除零错误
        total_votes = votes['real'] + votes['fake']
        if total_votes == 0:
            return "分析中..."
            
        # 计算统计信息
        real_ratio = votes['real'] / total_votes
        
        # 分析结果
        if real_ratio > 0.7:
            analysis = f"视频可能是真实的 (真实率: {real_ratio * 100:.1f}%)"
        elif real_ratio < 0.3:
            analysis = f"视频可能是AI生成的 (AI生成率: {(1 - real_ratio) * 100:.1f}%)"
        else:
            analysis = f"视频真伪难以确定\n真实率: {real_ratio * 100:.1f}%"
            
        return analysis

    def toggle_video(self):
        if hasattr(self, 'video_playing'):
            self.video_playing = not self.video_playing
            if self.video_playing:
                self.timer.start(30)
                self.play_pause_btn.setText("暂停")
            else:
                self.timer.stop()
                self.play_pause_btn.setText("播放")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ImageRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())