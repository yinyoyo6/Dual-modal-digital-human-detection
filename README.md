# 双模态数字人检测系统 (Dual-modal Digital Human Detection)

## 项目介绍

本项目是一个基于双模态（视频和音频）的数字人检测系统，旨在识别和区分真实人类与AI生成的数字人内容。系统通过分析视频帧的视觉特征和音频的声学特征，综合判断内容的真伪，为应对日益增长的AI生成内容带来的挑战提供解决方案。
![image](https://github.com/user-attachments/assets/ae8f2be3-9e98-4675-a69f-57a212f69f5d)


## 功能特点

- **双模态分析**：同时分析视频和音频特征，提高检测准确率
- **实时检测**：支持实时视频流分析和静态文件分析
- **可视化界面**：基于PyQt5的用户友好界面，便于操作和结果展示
- **异常特征标记**：自动标记并展示可疑的AI生成特征
- **结果历史记录**：保存分析历史，便于后续查看和比较
- **多模型支持**：集成多种深度学习模型，包括VGG、ResNet、DenseNet等

## 技术架构

### 视频分析模块

- 基于深度学习的视频帧特征提取
- 帧间特征一致性分析
- 面部特征异常检测
- 光影、纹理不自然性分析

### 音频分析模块

- 基于XLSR模型的音频特征提取
- 声音频谱异常检测
- 语音节奏和自然度分析

### 核心技术

- PyTorch深度学习框架
- OpenCV图像处理
- PyQt5图形界面
- 音频处理库（SoundFile, Librosa等）

## 安装说明

### 环境要求

- Python 3.7+
- CUDA支持（推荐但非必需）

### 依赖安装

```bash
# 使用清华镜像源加速安装
pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖包括：
- PyTorch (根据CUDA版本选择合适的安装命令)
- NumPy
- OpenCV
- PyQt5
- SoundFile
- Librosa
- Matplotlib

## 使用方法

### 启动应用

```bash
python main.py
```

### 使用流程

1. 在应用界面点击左侧区域选择要分析的图片或视频文件
2. 点击"开始识别"按钮进行内容分析
3. 系统将自动处理并显示分析结果，包括真伪判断和可疑特征
4. 分析结果会自动保存到历史记录中

### 批量处理

对于需要批量处理的场景，可以使用命令行模式：

```bash
python run_test.py --input_dir [输入目录] --output_dir [输出目录]
```

## 模型说明

本项目集成了多种深度学习模型用于特征提取和分类：

- **视频分析**：使用VGG、ResNet50、DenseNet121等模型进行视频帧分析
- **音频分析**：使用基于XLSR的特征提取器进行音频分析

模型文件存放在项目根目录和models目录下，首次运行时会自动加载。

## 项目结构

```
├── main.py           # 主程序入口
├── models/           # 模型定义
│   ├── vgg.py        # VGG模型
│   ├── resnet.py     # ResNet模型
│   └── ...           # 其他模型
├── log/              # 日志和临时文件
│   ├── results/      # 分析结果
│   ├── mp3_history/  # 音频文件
│   └── temp_frames/  # 临时视频帧
├── data/             # 训练数据
│   ├── REAL/         # 真实样本
│   └── FAKE/         # 伪造样本
├── 训练.py           # 模型训练脚本
└── requirement.txt   # 依赖包列表
```

## 贡献者

- [yinyoyo6](https://github.com/yinyoyo6)

## 联系方式

- Email: 1478924375@qq.com

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

感谢所有为本项目提供支持和贡献的个人和组织。
