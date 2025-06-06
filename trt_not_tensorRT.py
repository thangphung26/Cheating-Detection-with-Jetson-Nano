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

'''
img is PIL format
'''
def draw_keypoints(img, key):
    thickness = 5
    w, h = img.size
    draw = PIL.ImageDraw.Draw(img)
    #draw Rankle -> RKnee (16-> 14)
    if all(key[16]) and all(key[14]):
        draw.line([ int(key[16][2] * w), int(key[16][1] * h), int(key[14][2] * w), int(key[14][1] * h)],width = thickness, fill=(51,51,204))
    #draw RKnee -> Rhip (14-> 12)
    if all(key[14]) and all(key[12]):
        draw.line([ int(key[14][2] * w), int(key[14][1] * h), int(key[12][2] * w), int(key[12][1] * h)],width = thickness, fill=(51,51,204))
    #draw Rhip -> Lhip (12-> 11)
    if all(key[12]) and all(key[11]):
        draw.line([ int(key[12][2] * w), int(key[12][1] * h), int(key[11][2] * w), int(key[11][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lhip -> Lknee (11-> 13)
    if all(key[11]) and all(key[13]):
        draw.line([ int(key[11][2] * w), int(key[11][1] * h), int(key[13][2] * w), int(key[13][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lknee -> Lankle (13-> 15)
    if all(key[13]) and all(key[15]):
        draw.line([ int(key[13][2] * w), int(key[13][1] * h), int(key[15][2] * w), int(key[15][1] * h)],width = thickness, fill=(51,51,204))

    #draw Rwrist -> Relbow (10-> 8)
    if all(key[10]) and all(key[8]):
        draw.line([ int(key[10][2] * w), int(key[10][1] * h), int(key[8][2] * w), int(key[8][1] * h)],width = thickness, fill=(255,255,51))
    #draw Relbow -> Rshoulder (8-> 6)
    if all(key[8]) and all(key[6]):
        draw.line([ int(key[8][2] * w), int(key[8][1] * h), int(key[6][2] * w), int(key[6][1] * h)],width = thickness, fill=(255,255,51))
    #draw Rshoulder -> Lshoulder (6-> 5)
    if all(key[6]) and all(key[5]):
        draw.line([ int(key[6][2] * w), int(key[6][1] * h), int(key[5][2] * w), int(key[5][1] * h)],width = thickness, fill=(255,255,0))
    #draw Lshoulder -> Lelbow (5-> 7)
    if all(key[5]) and all(key[7]):
        draw.line([ int(key[5][2] * w), int(key[5][1] * h), int(key[7][2] * w), int(key[7][1] * h)],width = thickness, fill=(51,255,51))
    #draw Lelbow -> Lwrist (7-> 9)
    if all(key[7]) and all(key[9]):
        draw.line([ int(key[7][2] * w), int(key[7][1] * h), int(key[9][2] * w), int(key[9][1] * h)],width = thickness, fill=(51,255,51))

    #draw Rshoulder -> RHip (6-> 12)
    if all(key[6]) and all(key[12]):
        draw.line([ int(key[6][2] * w), int(key[6][1] * h), int(key[12][2] * w), int(key[12][1] * h)],width = thickness, fill=(153,0,51))
    #draw Lshoulder -> LHip (5-> 11)
    if all(key[5]) and all(key[11]):
        draw.line([ int(key[5][2] * w), int(key[5][1] * h), int(key[11][2] * w), int(key[11][1] * h)],width = thickness, fill=(153,0,51))


    #draw nose -> Reye (0-> 2)
    if all(key[0][1:]) and all(key[2]):
        draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[2][2] * w), int(key[2][1] * h)],width = thickness, fill=(219,0,219))

    #draw Reye -> Rear (2-> 4)
    if all(key[2]) and all(key[4]):
        draw.line([ int(key[2][2] * w), int(key[2][1] * h), int(key[4][2] * w), int(key[4][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> Leye (0-> 1)
    if all(key[0][1:]) and all(key[1]):
        draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[1][2] * w), int(key[1][1] * h)],width = thickness, fill=(219,0,219))

    #draw Leye -> Lear (1-> 3)
    if all(key[1]) and all(key[3]):
        draw.line([ int(key[1][2] * w), int(key[1][1] * h), int(key[3][2] * w), int(key[3][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> neck (0-> 17)
    if all(key[0][1:]) and all(key[17]):
        draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[17][2] * w), int(key[17][1] * h)],width = thickness, fill=(255,255,0))
    return img

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
            print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            print('index:%d : None'%(j) )
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

def save_keypoints_to_csv(image_name, humans, peaks, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)


        for i in range(len(humans[0])):
            human = humans[0][i]
            keypoints = []
            for j in range(human.shape[0]):
                k = int(human[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    keypoints.append((j, peak[0], peak[1])) 
                else:
                    keypoints.append((j, None, None)) 
            if sum(1 for kp in keypoints if kp[1] is not None) < 3:
                continue
            writer.writerow([f'{kp[1]},{kp[2]}' for kp in keypoints])

def execute(img):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        print("Human index:%d "%( i ))
        get_keypoint(objects, i, peaks)
    print("Human count:%d len:%d "%(counts[0], len(counts)))
    print('===== Net FPS :%f ====='%( 1 / (end - start)))
    draw_objects(img, counts, objects, peaks)
    return img

'''
Draw to original image
'''
def execute_2(image_name, img, org, csv_file):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    save_keypoints_to_csv(image_name, objects, peaks, csv_file)

    for i in range(counts[0]):
        print("Human index:%d "%( i ))
        kpoint = get_keypoint(objects, i, peaks)
        #print(kpoint)
        org = draw_keypoints(org, kpoint)
    print("Human count:%d len:%d "%(counts[0], len(counts)))
    print('===== Net FPS :%f ====='%( 1 / (end - start)))
    return org


parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--image', type=str, default='/home/thangphung/Downloads/train', help='Directory of images to process')
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


# data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
# if os.path.exists(OPTIMIZED_MODEL) == False:
#     model.load_state_dict(torch.load(MODEL_WEIGHTS))
#     model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
#     torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.eval()
# t0 = time.time()
# torch.cuda.current_stream().synchronize()
# for i in range(50):
#     y = model(data)
# torch.cuda.current_stream().synchronize()
# t1 = time.time()

# print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

csv_file_path = '/home/thangphung/Downloads/results/keypoints_results.csv'

for filename in os.listdir(args.image):
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        img_path = os.path.join(args.image, filename)
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #pilimg = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        orgimg = pilimg.copy()

        image = cv2.resize(src, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        parse_objects = ParseObjects(topology)
        draw_objects = DrawObjects(topology)

        img = image.copy()
        pilimg = execute_2(filename, img, orgimg, csv_file_path)

        output_dir = '/home/thangphung/Downloads/results'
        output_path = os.path.join(output_dir, f'{args.model}_{filename}')
        pilimg.save(output_path)
        print(f'Saved: {output_path}')


       



