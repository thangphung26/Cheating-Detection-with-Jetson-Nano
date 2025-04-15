import torch
import torch2trt
from torch2trt import TRTModule
import os
import trt_pose


HEIGHT = 224
WIDTH = 224
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


if not os.path.exists(OPTIMIZED_MODEL):
    print(f"Optimized model file '{OPTIMIZED_MODEL}' already exists.")
    
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    print("Model weights loaded successfully.")
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    print("Model converted to TensorRT successfully.")
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
