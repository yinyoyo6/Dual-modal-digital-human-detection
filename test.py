import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'None'}")