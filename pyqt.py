import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QFrame,
                           QScrollArea, QWidget, QProgressBar)
from PyQt5.QtGui import QPixmap, QFont, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QDateTime
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
        self.frame_results = []
        self.frame_window = 30
        self.result_threshold = 0.7
        self.history_results = []  # 存储历史识别结果
        self.max_history = 5  # 最多保存5条历史记录
        self.total_frames = 0  # 总帧数
        self.processed_frames = 0  # 已处理帧数
        self.is_analyzing = False  # 添加标志来表示是否正在进行视频分析
        self.stats_count = {
            'image': {'real': 0, 'fake': 0},
            'video': {'real': 0, 'fake': 0}
        }  # 分别统计图片和视频的检测结果
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
        """)
        self.init_ui()

    def init_ui(self):
        # Set up the main layout with margins
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left side for loading images
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.StyledPanel)
        left_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #dcdcdc;
            }
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Image display area
        self.image_label = QLabel("点击此处选择图片或视频")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 4px;
                color: #6c757d;
                font-size: 16px;
            }
            QLabel:hover {
                background-color: #e9ecef;
                border-color: #6c757d;
            }
        """)
        self.image_label.mousePressEvent = self.load_file
        left_layout.addWidget(self.image_label)

        # Video controls
        video_controls = QHBoxLayout()
        self.play_pause_btn = QPushButton("暂停")
        self.play_pause_btn.setFixedWidth(100)
        self.play_pause_btn.clicked.connect(self.toggle_video)
        self.play_pause_btn.setVisible(False)
        video_controls.addStretch()
        video_controls.addWidget(self.play_pause_btn)
        video_controls.addStretch()
        left_layout.addLayout(video_controls)

        # Middle section with recognition button
        middle_frame = QFrame()
        middle_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #dcdcdc;
            }
        """)
        middle_layout = QVBoxLayout(middle_frame)
        middle_layout.setContentsMargins(10, 10, 10, 10)

        recognize_button = QPushButton("开始识别")
        recognize_button.setFixedSize(120, 50)
        recognize_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        recognize_button.clicked.connect(self.recognize_image)
        middle_layout.addStretch(1)
        middle_layout.addWidget(recognize_button, alignment=Qt.AlignCenter)
        middle_layout.addStretch(1)

        # Right side for results
        right_frame = QFrame()
        right_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #dcdcdc;
            }
        """)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)

        # 当前结果区域
        current_result_frame = QFrame()
        current_result_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        current_result_layout = QVBoxLayout(current_result_frame)
        
        result_title = QLabel("识别结果")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3; margin-bottom: 10px;")
        current_result_layout.addWidget(result_title)

        self.result_label = QLabel("等待识别...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #495057;
                padding: 10px;
            }
        """)
        current_result_layout.addWidget(self.result_label)
        right_layout.addWidget(current_result_frame)

        # 分割线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #dee2e6;")
        right_layout.addWidget(separator)

        # 历史记录区域
        history_frame = QFrame()
        history_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 10px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a8a8a8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        history_layout = QVBoxLayout(history_frame)
        history_layout.setSpacing(5)
        
        history_title = QLabel("历史记录")
        history_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        history_layout.addWidget(history_title)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 创建滚动区域的内容容器
        scroll_content = QWidget()
        self.history_list = QVBoxLayout(scroll_content)
        self.history_list.setSpacing(5)
        self.history_list.setContentsMargins(0, 0, 0, 0)
        self.history_list.addStretch()
        
        scroll_area.setWidget(scroll_content)
        history_layout.addWidget(scroll_area)
        right_layout.addWidget(history_frame)

        # 统计信息区域
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        
        stats_title = QLabel("统计信息")
        stats_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        stats_layout.addWidget(stats_title)

        self.stats_label = QLabel("实时处理数据：\n- 真实图片：0次\n- AI生成：0次")
        self.stats_label.setStyleSheet("color: #495057;")
        stats_layout.addWidget(self.stats_label)
        
        right_layout.addWidget(stats_frame)
        right_layout.addStretch()

        # 添加进度条
        self.progress_frame = QFrame()
        self.progress_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 10px;
            }
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        progress_title = QLabel("视频处理进度")
        progress_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        progress_layout.addWidget(progress_title)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_frame.setVisible(False)
        right_layout.insertWidget(1, self.progress_frame)  # 在结果标签后面插入进度条

        # Add all frames to main layout
        main_layout.addWidget(left_frame, 4)
        main_layout.addWidget(middle_frame, 1)
        main_layout.addWidget(right_frame, 3)

        self.setWindowTitle("真 识")
        self.setGeometry(100, 100, 1200, 600)
        self.setMinimumSize(900, 500)

    def load_file(self, event):
        options = QFileDialog.Options()
        file_filter = "Media Files (*.png *.jpg *.bmp *.gif *.mp4 *.avi *.mov);;Images (*.png *.jpg *.bmp *.gif);;Videos (*.mp4 *.avi *.mov)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter, options=options)
        
        if file_name:
            # 重置进度和状态
            self.processed_frames = 0
            self.total_frames = 0
            self.progress_bar.setValue(0)
            self.is_analyzing = False
            self.frame_results.clear()
            
            # 如果之前有视频在播放，先停止并清理资源
            if hasattr(self, 'cap') and self.cap is not None:
                self.timer.stop()
                self.cap.release()
                self.cap = None
                self.video_playing = False
                self.play_pause_btn.setVisible(False)
            
            self.file_name = file_name
            self.is_video = self.file_name.lower().endswith(('.mp4', '.avi', '.mov'))
            
            if self.is_video:
                self.cap = cv2.VideoCapture(self.file_name)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(30)
                self.video_playing = True
                self.play_pause_btn.setVisible(True)
                self.play_pause_btn.setText("暂停")
                self.progress_frame.setVisible(True)
                # 显示第一帧但不开始分析
                ret, frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (400, 300))
                    h, w, ch = frame_resized.shape
                    qt_image = QImage(frame_resized.data, w, h, ch * w, QImage.Format_RGB888)
                    self.image_label.setPixmap(QPixmap.fromImage(qt_image))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.result_label.setText("点击\"开始识别\"按钮开始分析视频")
            else:
                self.progress_frame.setVisible(False)
                pixmap = QPixmap(self.file_name)
                pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                self.result_label.setText("点击\"开始识别\"按钮开始分析图片")
                
            self.image_label.setAlignment(Qt.AlignCenter)

    def update_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # 只有在分析模式下才处理帧
            if self.is_analyzing:
                analysis, current_score, is_last_frame = self.process_video_frame(frame)
                
                # 更新进度条
                progress = int((self.processed_frames / self.total_frames) * 100)
                self.progress_bar.setValue(progress)
                
                # 更新结果显示
                if analysis:
                    if is_last_frame:
                        # 视频分析完成，显示最终结果并暂停
                        final_analysis = self.get_final_video_analysis()
                        self.result_label.setText(final_analysis)
                        self.toggle_video()  # 暂停视频播放
                        # 将最终结果添加到历史记录
                        self.update_history("视频分析", final_analysis)
                        self.is_analyzing = False
                    else:
                        self.result_label.setText(analysis)
            
            # 更新显示画面
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (400, 300))
            h, w, ch = frame_resized.shape
            qt_image = QImage(frame_resized.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        else:
            if self.is_analyzing:
                # 视频播放完毕且在分析模式下，显示最终结果
                final_analysis = self.get_final_video_analysis()
                self.result_label.setText(final_analysis)
                self.toggle_video()  # 暂停视频播放
                # 将最终结果添加到历史记录
                self.update_history("视频分析", final_analysis)
                self.is_analyzing = False
            # 重置视频到开始
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.processed_frames = 0

    def recognize_image(self):
        try:
            if not hasattr(self, 'file_name') or not self.file_name:
                self.result_label.setText("请先选择文件")
                return
                
            if hasattr(self, 'is_video') and self.is_video:
                # 开始视频分析
                self.is_analyzing = True
                self.processed_frames = 0
                self.frame_results.clear()
                self.progress_bar.setValue(0)
                # 确保从头开始分析
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.result_label.setText("开始视频分析...")
                return
                
            # 图片处理逻辑保持不变
            input_image = get_imageNdarray(self.file_name)
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
                
                # 更新历史记录
                self.update_history("图片分析", result_text)
                
        except Exception as e:
            self.result_label.setText(f"识别出错: {str(e)}")

    def process_video_frame(self, frame):
        try:
            if not hasattr(self, 'total_frames') or self.total_frames == 0:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                result = classes[predicted_idx].lower()
                
                self.frame_results.append((result, score))
                self.processed_frames += 1
                
                # 判断是否是最后一帧
                is_last_frame = self.processed_frames >= self.total_frames
                
                analysis = self.analyze_current_frame(show_progress=True)
                return analysis, score, is_last_frame
                
        except Exception as e:
            print(f"帧处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0, False

    def analyze_current_frame(self, show_progress=False):
        if not self.frame_results:
            return "等待分析..."
        
        # 只分析最近30帧的结果用于实时显示
        recent_results = self.frame_results[-30:]
        
        votes = {'real': 0, 'fake': 0}
        for result, score in recent_results:
            result = result.lower()
            if result in votes:
                votes[result] += score
        
        total_votes = votes['real'] + votes['fake']
        if total_votes == 0:
            return "分析中..."
            
        real_ratio = votes['real'] / total_votes
        progress = (self.processed_frames / self.total_frames) * 100 if show_progress else 0
        
        if real_ratio > 0.7:
            return f"当前片段: 真实 (真实率: {real_ratio * 100:.1f}%)\n处理进度: {progress:.1f}%"
        elif real_ratio < 0.3:
            return f"当前片段: AI生成 (AI生成率: {(1 - real_ratio) * 100:.1f}%)\n处理进度: {progress:.1f}%"
        else:
            return f"当前片段: 难以确定\n真实率: {real_ratio * 100:.1f}%\n处理进度: {progress:.1f}%"

    def get_final_video_analysis(self):
        """分析整个视频的结果"""
        if not self.frame_results:
            return "没有可分析的数据"

        total_real_score = 0
        total_fake_score = 0
        frame_count = len(self.frame_results)
        
        # 计算所有帧的累积分数
        for result, score in self.frame_results:
            if result == 'real':
                total_real_score += score
            elif result == 'fake':
                total_fake_score += score

        # 计算整体的真实率
        total_score = total_real_score + total_fake_score
        if total_score == 0:
            return "视频分析失败"

        real_ratio = total_real_score / total_score
        
        # 生成详细的分析报告
        report = "视频分析完成\n"
        report += f"总帧数: {frame_count}\n"
        report += f"整体真实率: {real_ratio * 100:.1f}%\n"
        report += f"整体AI生成率: {(1 - real_ratio) * 100:.1f}%\n\n"
        
        # 最终结论
        if real_ratio > 0.7:
            report += "最终判定: 该视频很可能是真实的"
        elif real_ratio < 0.3:
            report += "最终判定: 该视频很可能是AI生成的"
        else:
            report += "最终判定: 该视频真伪难以确定，建议进一步分析"
            
        return report

    def update_history(self, analysis_type, result):
        """更新历史记录
        @param analysis_type: 分析类型（"图片分析"或"视频分析"）
        @param result: 分析结果文本
        """
        # 创建新的历史记录标签
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        history_item = QLabel(f"[{timestamp}] {analysis_type}:\n{result}")
        history_item.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                margin: 2px;
                color: #495057;
            }
        """)
        history_item.setWordWrap(True)
        
        # 添加新的历史记录到列表开头
        if self.history_list.count() >= self.max_history:
            # 删除最后一个项目
            item = self.history_list.takeAt(self.history_list.count() - 1)
            if item.widget():
                item.widget().deleteLater()
        
        self.history_list.insertWidget(0, history_item)
        
        # 更新统计信息
        if "视频分析" in analysis_type:
            # 从结果文本中提取最终判定结果
            if "很可能是真实的" in result:
                self.update_statistics('real', 'video')
            elif "很可能是AI生成的" in result:
                self.update_statistics('fake', 'video')
        else:  # 图片分析
            result = result.split('\n')[0].split(': ')[1].lower()  # 提取结果类型
            self.update_statistics(result, 'image')

    def update_statistics(self, result, media_type='image'):
        """更新统计数据
        @param result: 检测结果 ('real' 或 'fake')
        @param media_type: 媒体类型 ('image' 或 'video')
        """
        result = result.lower()
        if result in ['real', 'fake']:
            self.stats_count[media_type][result] += 1
            
        stats_text = "实时处理数据：\n"
        stats_text += f"图片检测：\n- 真实：{self.stats_count['image']['real']}次\n- AI生成：{self.stats_count['image']['fake']}次\n"
        stats_text += f"视频检测：\n- 真实：{self.stats_count['video']['real']}次\n- AI生成：{self.stats_count['video']['fake']}次"
        self.stats_label.setText(stats_text)

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