import torch
print(torch.cuda.is_available())  # Check if GPU is available
print(torch.cuda.get_device_name(0))  # Get GPU name
print(torch.cuda.memory_summary())  # Get memory usage summary
