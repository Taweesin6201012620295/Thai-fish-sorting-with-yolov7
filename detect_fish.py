import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math

from edit_bboxes import edit_boxes, color_class

from models.experimental import attempt_load 
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging, xyxy2xywh, increment_path
from utils.torch_utils import select_device, TracedModel

def detect_fish(source, weights, view_img, save_txt, imgsz, trace, save_image, project, name, exist_ok,
                device, augment, conf_thres, iou_thres, classes, agnostic_nms, save_conf):
    
    save_img = save_image #and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() #or source.endswith('.txt') or source.lower().startswith( ('rtsp://', 'rtmp://', 'http://', 'https://') )
    
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
                        print(f'{names[int(name_class)]} {conf:.2f}')
                        line = f"{int(name_class)} {x_center} {y_center} {width} {height} {conf}"
                        #print(line)
                        
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
           
    detect_fish(
                source="0",
                weights=["yolov7_4000_6.pt"],
                view_img = False,
                save_conf=True,  
                save_txt = True,
                save_image = True,
                device='cpu',
                conf_thres=0.25,
                imgsz = 640,
                trace = False,
                project = 'runs/detect',
                name = "ex",
                exist_ok=False,
                augment=False,
                iou_thres=0.45,
                classes=None,
                agnostic_nms=False
                )
    