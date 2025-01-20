import torch

# PyTorch sürümü
print("PyTorch Version:", torch.__version__)

# CUDA erişimi
print("CUDA Available:", torch.cuda.is_available())

# Aktif GPU cihazı
if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
