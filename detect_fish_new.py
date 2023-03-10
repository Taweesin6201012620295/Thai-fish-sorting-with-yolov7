import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math

from models.experimental import attempt_load 
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging, xyxy2xywh, increment_path
from utils.torch_utils import select_device, TracedModel

from hd_nanpy import call_arduino
from hd_nanpy import convenyor_run
from hd_nanpy import convenyor_stop
from threading import Thread
    


# check range
def in_range(new_val, old_val):
    new_val, old_val = float(new_val), float(old_val)
    n = 0.05  
    if new_val >= old_val-n and new_val <= old_val+n:
        return True
    else:
        return False
    

# check same location    
def check_range(j,temp):
    x = in_range(j[1],temp[1])
    y = in_range(j[2],temp[2])
    w = in_range(j[3],temp[3])
    h = in_range(j[4],temp[4])
    #print("xywh : ",x,y,w,h)
    
    if x and y and w and h :
        #print(x,y,w,h)
        if j[5] >= temp[5]:
            #print(j[5], "max")
            return j
        elif temp[5] >= j[5]:
            #print(temp[5], "max")
            return temp
    else:
        #print("return_check_range : False")
        return False
 
# edit bboxes
def edit_boxes(arr):
    
    arr = np.array(arr)
    array_sort = arr[arr[:,1].argsort()]

    #print("array_sort : \n", array_sort)
    #print(len(array_sort))
    get_list = []
    state = False
    check_old = None

    for i, j in enumerate(array_sort):
        
        if i==0:
            temp = np.array(array_sort[i])
        if i>0  and not(np.array_equal(j, temp)):
            #print('temp     is ', temp)
            #print('new_temp is ', j)
            check = check_range(j,temp)
            #print('check : ', check)
            #print('check_old :', check_old)
            
            if type(check) == np.ndarray:
                if len(get_list) != 0:
                    if type(check_old) == np.ndarray and not np.array_equal(check, check_old):
                        #print("choose max conf >> pop1")
                        # check after list have conf max same now
                        new_check = check_range(check, check_old)
                        get_list.pop()
                        get_list.append(new_check.tolist())
                    else:
                        #print("choose max conf >> pop2")
                        get_list.pop()
                        get_list.append(check.tolist())
                else: # don't have anything in list
                    #print("choose max conf >> no_pop1")
                    get_list.append(check.tolist())
                
            else: # False
                if len(get_list) != 0:
                    #print("this's other box1 >> new_temp")
                    get_list.append(j.tolist())
                else:
                    #print("this's other box2 >> temp+new_temp")
                    get_list.append(temp.tolist())
                    get_list.append(j.tolist())
                    
            temp = j
            check_old = check
            #print("get_list : ", get_list)
        #print("-----------------------------------------------------------------")
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


def detect_fish(source, weights, view_img, save_txt, imgsz, trace, save_image, project, name, exist_ok,
                device, augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf):
    
    save_img = save_image #and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() #or source.endswith('.txt') or source.lower().startswith( ('rtsp://', 'rtmp://', 'http://', 'https://') )
    check_type = []
    
    # Directories
    if save_txt and not source.isnumeric():
        print("make dir save")
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    device = select_device(device) # my device type='cpu'
    
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
        print("webcam source <------------------------------------")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        print("image source <------------------------------------")


    # Get names and colors
    '''names = ['pod', 'ku_lare', 'see_kun', 'too', 'khang_pan', 'hang_lueang', 'sai_dang', 'sai_dum']'''
    names = model.module.names if hasattr(model, 'module') else model.names 

    # Run inference 
    if device.type != 'cpu':
        print("device != cpu")
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz # 640
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset: # path=['0'], img=[image], vid_cap=None
        # image small size
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup <<<<<< if "cpu" don't use
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

        # Process detections <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) # don't use

            p = Path(p)  # to Path windowspath='0'
            if save_img and save_txt and not source.isnumeric():
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh=[640,480,640,480]
            if len(det): # !!!!!!!
                #print("len det", len(det))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results text file  
                temp = []
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    ## edit label 
                    temp.append([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]) ,float(xywh[3]) ,float(conf)])
                #print("temp : ", temp)

                try:        
                    if len(temp) != 1:
                        temp = edit_boxes(temp)
                        #print('result1 : ', len(temp), temp)
                        temp = edit_boxes(temp)
                        #print('result2 : ', len(temp), temp)
                    else:
                        #print("temp : ", temp)
                        pass
                
                    for i in temp:
                        name_class, x_center, y_center, width, height, conf = i[0], i[1], i[2], i[3], i[4], i[5]
                        #print(f'{names[int(name_class)]} {conf:.2f}')
                        line = f"{int(name_class)} {x_center} {y_center} {width} {height} {conf}"
                        #print(line)
                        if len(check_type) <= 5:
                            check_type.append(names[int(name_class)])
                        print(check_type)
                        if len(check_type) >= 5:
                            fish_out = max(check_type, key=check_type.count)
                            print(fish_out)
                            call_arduino(fish_out)
                            check_type.clear()
                        
                        if save_txt and not source.isnumeric():
                            #print("save-txt")
                            if save_conf == False:
                                line = f"{int(name_class)} {x_center} {y_center} {width} {height}"
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(line + '\n')

                        if save_img:
                            # Add bbox to image
                            #print("add-label-image")
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
                            tf = max(tl - 1, 0.5)  # font thickness
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                            # Stream results
                            if not source.isnumeric() and save_img:
                                #print("save-image")
                                cv2.imwrite(save_path, im0)
                    
                    #print("Result: ", temp) # array label [class, x, y, w,h, confident] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                except:pass  

                print("---------------------------------------------------")
                

            if view_img and source.isnumeric():
                #print("show-real-time")
                cv2.imshow("show fish", im0) 
                cv2.waitKey(1)  # 1 millisecond

        
if __name__ == '__main__':
    
    thr1 = Thread(target=detect_fish, args=["0", ["yolov7_3200.pt"], 
                False, True, 640, False, True, 'runs/detect', 'name', False,
                'cpu', False, 0.25, 0.45, None, False, True])
    thr2 = Thread(target=convenyor_run)

    thr2.start()
    thr1.start()
