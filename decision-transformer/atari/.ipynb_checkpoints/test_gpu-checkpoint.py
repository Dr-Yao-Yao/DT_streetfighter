import torch
import tensorflow as tf
flag = torch.cuda.is_available()
print("torch", flag)
print("tf",tf.config.list_physical_devices('GPU'))
