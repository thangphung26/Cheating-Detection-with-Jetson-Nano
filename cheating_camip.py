from CheatDetection.__init__ import CheatDetection
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image, PIL.ImageDraw
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from PIL import Image
import argparse
import os.path
import csv
import os
import subprocess
from CheatDetection.__init__ import CheatDetection




'''
hnum: 0 based human index
kpoint : index + keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height)
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None'%(j) )
    return kpoint

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

'''
Draw to inference (small)image
'''




'''
Draw to original image
'''
def execute_2(img, src, t):
    
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    fps = 1.0 / (time.time() - t)
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    #save_keypoints_to_csv(image_name, objects, peaks, csv_file)
    keypoint = []
    
    for i in range(counts[0]):
        
        kpoint = get_keypoint(objects, i, peaks)
        
        keypoint = [kpoint]
        #print(keypoint)
        cheating_keypoints, is_cheating = detector.DetectCheat(keypoint)
        if is_cheating:
            keypoint_01 = (cheating_keypoints[0][0][1], cheating_keypoints[0][0][0])  # (x, y)
            
            break
        

    print("FPS:%f "%(fps))
    # cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    print("Writing frame", count)
    # out_video.write(src)

    return keypoint_01
parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256


data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


cap = cv2.VideoCapture('/home/thangphung/trt_pose/tasks/human_pose/test3.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_video = cv2.VideoWriter('/home/thangphung/trt_pose/tasks/human_pose/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
count = 0

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0


while cap.isOpened() and count<500:
    t = time.time()
    ret, frame = cap.read()
    if not ret:
        break
        #pilimg = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    orgimg = pilimg.copy()

    image = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    detector = CheatDetection()
    img = image.copy()
    keypoint_01 = execute_2(img, frame, t)

    
    if keypoint_01:
        print(f"Toa do phat hien: {keypoint_01}")
        
    else:
        print("Khong phat hien gian lan!")
    count += 1
    
          
            
cap.release()
out_video.release()
cv2.destroyAllWindows()     
    