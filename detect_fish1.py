import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math
from models.experimental import attempt_load 
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.torch_utils import select_device, TracedModel
import hd_nanpy

def in_range(new_val, old_val):
    new_val, old_val = float(new_val), float(old_val)
    n = 0.1 
    if new_val >= old_val-n and new_val <= old_val+n:
        return True # in range
    else:
        return False # out of range

def check_range(j,temp):
    cl = False if j[0] == temp[0] else True
    x = in_range(j[1],temp[1])
    y = in_range(j[2],temp[2])
    w = in_range(j[3],temp[3])
    h = in_range(j[4],temp[4])
    if x and y and w and h :
        # check confident
        if j[5] >= temp[5]:
            return j
        elif temp[5] >= j[5]:
            return temp
    elif cl == True and not(x and y and w and h) :
        return True
    else:
        return False
        

def edit_boxes(arr):
    arr = np.array(arr)
    array_sort = arr[arr[:,1].argsort()]
    get_list = []
    state = False

    for i, j in enumerate(array_sort):
        if i==0:
            temp = np.array(array_sort[i])

        if i>0  and not(np.array_equal(j, temp)):

            # check same box
            check = check_range(j,temp)
            if check is True :
                if len(get_list) != 0:
                    get_list.append(j.tolist())
                    state = False
                else:
                    get_list.append(temp.tolist())
                    get_list.append(j.tolist())
                    state = False

            elif type(check) == np.ndarray:
                if state == False :
                    if len(get_list) != 0:
                        get_list.pop()
                        get_list.append(check.tolist())
                    else:
                        get_list.append(check.tolist())
                elif state == True :
                    get_list.append(check.tolist())
                state = True

            else: # check = False
                if len(get_list) == 0:
                    get_list.append(temp.tolist())
                    get_list.append(j.tolist())
                else:
                    get_list.append(j.tolist())
            temp = j
    return get_list
    
def color_class(index_class):
    match index_class:
        case 0:
            return (0,0,0) # pod is black
        case 1:
            return (255,0,0) # ku_lare is blue
        case 2:
            return (255,191,0) # see_kun is deep sky blue
        case 3:
            return (19,69,139) # too is saddle brown
        case 4:
            return (127,20,255) # khang_pan is deep pink
        case 5:
            return (0,140,255) # hang_lueang is orange
        case 6:
            return (0,0,255) # sai_dang is red
        case 7:
            return (128,0,128) # sai_dum is purple


def detect_fish( 
    source="0",
    weights=["yolov7_3200.pt"],
    view_img = False,
    save_txt = False,
    imgsz = 640,
    trace = False,
    nosave = False,
    project = 'runs/detect',
    name = "ex",
    exist_ok=False,
    device='',
    augment=False,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic_nms=False,
    save_conf=False):

    check_type = []

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric()
    
    # Directories
    if nosave and save_txt is True:
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        

    # Initialize
    device = select_device(device) # checkdevice >> my device type='cpu'
    
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model 
    model = attempt_load(weights, map_location=device)  # load FP32 model  <<< edit and use file_test.py
    stride = int(model.stride.max())  # model stride : 32
    imgsz = check_img_size(imgsz, s=stride)  # check img_size : 640


    if trace: # Trace : False
        model = TracedModel(model, device, imgsz)

    if half: 
        model.half()  # to FP16

    # Webcam
    view_img = check_imshow() # view_img : True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    ''' dataset : fps=30.0 img_size=640 imgs=[] mode=stream rect=True sources=['0'] stride=32 '''
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)


    # Get names and colors
    '''names = ['pod', 'ku_lare', 'see_kun', 'too', 'khang_pan', 'hang_lueang', 'sai_dang', 'sai_dum']'''
    names = model.module.names if hasattr(model, 'module') else model.names 
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference 
    if device.type != 'cpu':
        #print("device != cpu")
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz # 640
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset: # path=['0], img=[image], vid_cap=None
        # image small size
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup <<< if "cpu" don't use
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) # don't use

            p = Path(p)  # to Path windowspath='0'
            if nosave and save_txt is True:
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh=[640,480,640,480]

            if len(det): # detect have object in image
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results text file  
                temp = [] # get all label object
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    ## edit label 
                    temp.append([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]) ,float(xywh[3]) ,float(conf)])

                try:        
                    # check same label
                    edited = edit_boxes(temp)
                    edited2 = edit_boxes(edited)
                
                    for i in edited2:
                        name_class, x_center, y_center, width, height, conf = i[0], i[1], i[2], i[3], i[4], i[5]
                        line = f"{int(name_class)} {x_center} {y_center} {width} {height} {conf}"
                        print(f'{names[int(name_class)]} {conf:.2f}')
                        if len(check_type) <= 10:
                            check_type.append(names[int(name_class)])
                        if len(check_type) >= 10 or len(check_type) == 0:
                            hd_nanpy.convenyor_run
                        elif len(check_type) > 0 :
                            hd_nanpy.convenyor_stop

                        save_txt = False
                        save_conf = False
                        
                        if save_txt:
                            if save_conf == False:
                                line = f"{int(name_class)} {x_center} {y_center} {width} {height}"
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(line + '\n')
                            
                        if save_img:  # Add bbox to image
                            label = f'{names[int(name_class)]} {conf:.2f}'
                            line_thickness=5
                            # convert xywh to xyxy
                            h_img, w_img, channel = im0.shape 
                            x_min = math.ceil((x_center - width / 2) * w_img)
                            y_min = math.ceil((y_center - height /2) * h_img)
                            x_max = math.ceil((x_center + width /2) * w_img)
                            y_max = math.ceil((y_center + height /2) * h_img)
                            # write rectangle (box)
                            tl = line_thickness or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
                            c1,c2 = (x_min, y_min), (x_max, y_max)
                            color = color_class(int(name_class))
                            cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA) #change size 
                            # write label
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)     
                    #print("name:"+ names[int(name_class)])
                    print("---------------------------------------------------")
                
                except:pass  

            # save image
            if not source.isnumeric() and save_img: 
                cv2.imwrite(save_path, im0)

            # Stream results
            if view_img and source.isnumeric():
                cv2.imshow("show fish", im0) 
                cv2.waitKey(1)  # 1 millisecond
        print(check_type)
        if len(check_type) >= 10:
            fish_out = max(check_type, key=check_type.count)
            print(fish_out)
            hd_nanpy.call_arduino(fish_out)
            check_type = []
            
        
if __name__ == '__main__':
    detect_fish(source="0", #"0"com, "1"webcam
    weights=["yolov7_3200.pt"],
    view_img = False,
    save_txt = False,
    imgsz = 640,
    trace = False,
    nosave = False,
    project = 'runs/detect',
    name = "ex",
    exist_ok=False,
    device='',
    augment=False,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic_nms=False,
    save_conf=False)