from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import cv2

import torch, math
from pathlib import Path
import torch.backends.cudnn as cudnn
from edit_bboxes import edit_boxes, color_class
from models.experimental import attempt_load 
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.torch_utils import select_device, TracedModel

from threading import Thread
#from hd_nanpy import convenyor_run, convenyor_stop, call_arduino

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(951, 871)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setKerning(True)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("fish.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(110, 760, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.exit.setFont(font)
        self.exit.setStyleSheet("#exit{\n"
"border-radius: 10px;\n"
"    background-color: rgb(155, 215, 255);\n"
"}\n"
"\n"
"#exit:hover{\n"
"background-color: rgb(117, 177, 255);\n"
"}\n"
"\n"
"#exit:pressed{\n"
"background-color: rgb(139, 234, 255);\n"
"}")
        self.exit.setObjectName("exit")
        self.stop = QtWidgets.QPushButton(self.centralwidget)
        self.stop.setGeometry(QtCore.QRect(680, 760, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.stop.setFont(font)
        self.stop.setStyleSheet("#stop{\n"
"border-radius: 10px;\n"
"background-color: rgb(255, 206, 206);\n"
"}\n"
"\n"
"#stop:hover{\n"
"background-color: rgb(255, 156, 156);\n"
"}\n"
"\n"
"#stop:pressed{\n"
"background-color: rgb(255, 197, 221);\n"
"}")
        self.stop.setObjectName("stop")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(400, 760, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.start.setFont(font)
        self.start.setStyleSheet("#start{\n"
"border-radius: 10px;\n"
"    background-color: rgb(169, 255, 175);\n"
"}\n"
"\n"
"#start:hover{\n"
"background-color: rgb(117, 255, 149);\n"
"}\n"
"\n"
"#start:pressed{\n"
"background-color: rgb(186, 255, 160);\n"
"}")
        self.start.setObjectName("start")
        self.img_too = QtWidgets.QLabel(self.centralwidget)
        self.img_too.setGeometry(QtCore.QRect(700, 360, 180, 80))
        self.img_too.setText("")
        self.img_too.setPixmap(QtGui.QPixmap("img_fish/too.jpg"))
        self.img_too.setScaledContents(True)
        self.img_too.setObjectName("img_too")
        self.checkBox_hang_lueang = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_hang_lueang.setGeometry(QtCore.QRect(60, 290, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_hang_lueang.setFont(font)
        self.checkBox_hang_lueang.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_hang_lueang.setObjectName("checkBox_hang_lueang")
        self.bg_fish = QtWidgets.QLabel(self.centralwidget)
        self.bg_fish.setGeometry(QtCore.QRect(30, 40, 891, 681))
        self.bg_fish.setStyleSheet("border-radius: 10px;\n"
"background-color: rgb(244, 244, 244);\n"
"")
        self.bg_fish.setText("")
        self.bg_fish.setObjectName("bg_fish")
        self.img_sai_dum = QtWidgets.QLabel(self.centralwidget)
        self.img_sai_dum.setGeometry(QtCore.QRect(700, 180, 180, 80))
        self.img_sai_dum.setText("")
        self.img_sai_dum.setPixmap(QtGui.QPixmap("img_fish/sai_dum.jpg"))
        self.img_sai_dum.setScaledContents(True)
        self.img_sai_dum.setObjectName("img_sai_dum")
        self.img_see_kun = QtWidgets.QLabel(self.centralwidget)
        self.img_see_kun.setGeometry(QtCore.QRect(270, 360, 180, 80))
        self.img_see_kun.setText("")
        self.img_see_kun.setPixmap(QtGui.QPixmap("img_fish/see_kun.jpg"))
        self.img_see_kun.setScaledContents(True)
        self.img_see_kun.setObjectName("img_see_kun")
        self.img_ku_lare = QtWidgets.QLabel(self.centralwidget)
        self.img_ku_lare.setGeometry(QtCore.QRect(700, 90, 180, 80))
        self.img_ku_lare.setText("")
        self.img_ku_lare.setPixmap(QtGui.QPixmap("img_fish/ku_lare.jpg"))
        self.img_ku_lare.setScaledContents(True)
        self.img_ku_lare.setObjectName("img_ku_lare")
        self.checkBox_pod = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_pod.setGeometry(QtCore.QRect(60, 110, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_pod.setFont(font)
        self.checkBox_pod.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_pod.setObjectName("checkBox_pod")
        self.img_sai_dang = QtWidgets.QLabel(self.centralwidget)
        self.img_sai_dang.setGeometry(QtCore.QRect(270, 180, 180, 80))
        self.img_sai_dang.setText("")
        self.img_sai_dang.setPixmap(QtGui.QPixmap("img_fish/sai_dang.jpg"))
        self.img_sai_dang.setScaledContents(True)
        self.img_sai_dang.setObjectName("img_sai_dang")
        self.checkBox_sai_dang = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_sai_dang.setGeometry(QtCore.QRect(60, 200, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_sai_dang.setFont(font)
        self.checkBox_sai_dang.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_sai_dang.setObjectName("checkBox_sai_dang")
        self.img_fish_all = QtWidgets.QLabel(self.centralwidget)
        self.img_fish_all.setGeometry(QtCore.QRect(390, 490, 311, 181))
        self.img_fish_all.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.img_fish_all.setText("")
        self.img_fish_all.setPixmap(QtGui.QPixmap("img_fish/mix539.jpg"))
        self.img_fish_all.setScaledContents(True)
        self.img_fish_all.setObjectName("img_fish_all")
        self.img_pod = QtWidgets.QLabel(self.centralwidget)
        self.img_pod.setGeometry(QtCore.QRect(270, 90, 180, 80))
        self.img_pod.setText("")
        self.img_pod.setPixmap(QtGui.QPixmap("img_fish/pod.jpg"))
        self.img_pod.setScaledContents(True)
        self.img_pod.setObjectName("img_pod")
        self.img_hang_lueang = QtWidgets.QLabel(self.centralwidget)
        self.img_hang_lueang.setGeometry(QtCore.QRect(270, 270, 180, 80))
        self.img_hang_lueang.setText("")
        self.img_hang_lueang.setPixmap(QtGui.QPixmap("img_fish/hang_lueang.jpg"))
        self.img_hang_lueang.setScaledContents(True)
        self.img_hang_lueang.setObjectName("img_hang_lueang")
        self.checkBox_see_kun = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_see_kun.setGeometry(QtCore.QRect(60, 380, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_see_kun.setFont(font)
        self.checkBox_see_kun.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_see_kun.setObjectName("checkBox_see_kun")
        self.checkBox_all = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_all.setGeometry(QtCore.QRect(130, 490, 231, 40))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.checkBox_all.setFont(font)
        self.checkBox_all.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_all.setObjectName("checkBox_all")
        self.img_khang_pan = QtWidgets.QLabel(self.centralwidget)
        self.img_khang_pan.setGeometry(QtCore.QRect(700, 270, 180, 80))
        self.img_khang_pan.setText("")
        self.img_khang_pan.setPixmap(QtGui.QPixmap("img_fish/khang_pan.jpg"))
        self.img_khang_pan.setScaledContents(True)
        self.img_khang_pan.setObjectName("img_khang_pan")
        self.checkBox_ku_lare = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_ku_lare.setGeometry(QtCore.QRect(490, 110, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_ku_lare.setFont(font)
        self.checkBox_ku_lare.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_ku_lare.setObjectName("checkBox_ku_lare")
        self.checkBox_sai_dum = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_sai_dum.setGeometry(QtCore.QRect(490, 200, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_sai_dum.setFont(font)
        self.checkBox_sai_dum.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_sai_dum.setObjectName("checkBox_sai_dum")
        self.checkBox_khang_pan = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_khang_pan.setGeometry(QtCore.QRect(490, 290, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_khang_pan.setFont(font)
        self.checkBox_khang_pan.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_khang_pan.setObjectName("checkBox_khang_pan")
        self.checkBox_too = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_too.setGeometry(QtCore.QRect(490, 380, 190, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBox_too.setFont(font)
        self.checkBox_too.setStyleSheet("border-radius:5px;\n"
"background-color: rgb(235, 235, 235);")
        self.checkBox_too.setObjectName("checkBox_too")
        self.bg_fish.raise_()
        self.exit.raise_()
        self.stop.raise_()
        self.start.raise_()
        self.img_too.raise_()
        self.img_sai_dum.raise_()
        self.img_see_kun.raise_()
        self.img_ku_lare.raise_()
        self.checkBox_pod.raise_()
        self.img_sai_dang.raise_()
        self.checkBox_sai_dang.raise_()
        self.img_fish_all.raise_()
        self.img_pod.raise_()
        self.img_hang_lueang.raise_()
        self.checkBox_see_kun.raise_()
        self.checkBox_all.raise_()
        self.img_khang_pan.raise_()
        self.checkBox_ku_lare.raise_()
        self.checkBox_sai_dum.raise_()
        self.checkBox_khang_pan.raise_()
        self.checkBox_too.raise_()
        self.checkBox_hang_lueang.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.checkBox_all.stateChanged.connect(self.select_all)
        self.start.clicked.connect(self.start_work)
        self.stop.clicked.connect(self.stop_work)
        self.exit.clicked.connect(self.stop_exit)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fish sorting"))
        self.exit.setText(_translate("MainWindow", "exit"))
        self.stop.setText(_translate("MainWindow", "stop"))
        self.start.setText(_translate("MainWindow", "start"))
        self.checkBox_hang_lueang.setText(_translate("MainWindow", "ปลาหางเหลือง"))
        self.checkBox_pod.setText(_translate("MainWindow", "ปลาป๊อด"))
        self.checkBox_sai_dang.setText(_translate("MainWindow", "ปลาทรายแดง"))
        self.checkBox_see_kun.setText(_translate("MainWindow", "ปลาสีกุล"))
        self.checkBox_all.setText(_translate("MainWindow", "ปลา 8 สายพันธุ์"))
        self.checkBox_ku_lare.setText(_translate("MainWindow", "ปลากุแร"))
        self.checkBox_sai_dum.setText(_translate("MainWindow", "ปลาทรายดำ"))
        self.checkBox_khang_pan.setText(_translate("MainWindow", "ปลาข้างปาน"))
        self.checkBox_too.setText(_translate("MainWindow", "ปลาทู"))

    ##############################################################################################################
    def select_all(self, state):
        checkboxs = [self.checkBox_hang_lueang, self.checkBox_khang_pan, 
                     self.checkBox_pod, self.checkBox_ku_lare,
                     self.checkBox_see_kun, self.checkBox_too, 
                     self.checkBox_sai_dang, self.checkBox_sai_dum]
        for checkbox in checkboxs:
            checkbox.setCheckState(state)

    # type fish checked 
    def check(self):
        store_select = []
        self.state_fish = False
        if self.checkBox_pod.isChecked():
            store_select.append("pod")
            self.state_fish = True
        if self.checkBox_ku_lare.isChecked():
            store_select.append("ku_lare")
            self.state_fish = True
        if self.checkBox_see_kun.isChecked():
            store_select.append("see_kun")
            self.state_fish = True
        if self.checkBox_too.isChecked():
            store_select.append("too")
            self.state_fish = True
        if self.checkBox_khang_pan.isChecked():
            store_select.append("khang_pan")
            self.state_fish = True
        if self.checkBox_hang_lueang.isChecked():
            store_select.append("hang_lueang")            
            self.state_fish = True
        if self.checkBox_sai_dang.isChecked():
            store_select.append("sai_dang")
            self.state_fish = True
        if self.checkBox_sai_dum.isChecked():
            store_select.append("sai_dum")
        return store_select
        
    # start work after selected checkbox
    def start_work(self):
        self.check()
        if self.state_fish:
            self.start.hide()
            #run conveyor
            #convenyor_run()

            # detect with camera ----------------------------------------------------------------------------------------------------- #

            # setting
            source="0"
            weights=["yolov7_3200.pt"]
            view_img = False
            save_conf=False
            save_txt = True
            save_image = True
            device=''
            conf_thres=0.25
            imgsz = 640
            trace = False
            project = 'runs/detect'
            name = "ex"
            exist_ok=False
            augment=False
            iou_thres=0.45
            classes=None
            agnostic_nms=False

            save_img = save_image 
            webcam = source.isnumeric()
            check_type_fish = []
            no_change = 0
            selected_fish = self.check()

            # Directories
            if save_txt and not source.isnumeric():
                save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
                (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            # Initialize
            device = select_device(device) # my device type='cpu'
            half = device.type != 'cpu'  # half precision only supported on CUDA
            # Load model 
            model = attempt_load(weights, map_location=device)  # load FP32 model
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
            '''if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once'''
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
                # Process detections 
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
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # Write results text file  
                        temp = []
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            ## edit label 
                            temp.append([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]) ,float(xywh[3]) ,float(conf)])
                        try:        
                            if len(temp) != 1:
                                temp = edit_boxes(temp)
                                temp = edit_boxes(temp)
                            else:
                                pass
                            for i in temp:
                                name_class, x_center, y_center, width, height, conf = i[0], i[1], i[2], i[3], i[4], i[5]
                                line = f"{int(name_class)} {x_center} {y_center} {width} {height} {conf}"
                                # check detect type fish for open gate
                                check_type_fish.append(names[int(name_class)])
                                print(check_type_fish)

                                if save_txt and not source.isnumeric():
                                    if save_conf == False:
                                        line = f"{int(name_class)} {x_center} {y_center} {width} {height}"
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(line + '\n')
                                if save_img:
                                    # Add bbox to image
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
                                        cv2.imwrite(save_path, im0)
                                    self.show_label(label)
                        except:pass
                    else:
                        # not dectected fish --> call arduino
                        if len(check_type_fish) > 0:
                            no_change += 1
                            if no_change > 1:
                                fish_out = max(check_type_fish, key=check_type_fish.count)
                                print(fish_out)
                                #thr3 = Thread(target=call_arduino, args=[fish_out, selected_fish])
                                #thr3.start()
                                check_type_fish = []
                                no_change = 0

                    if view_img and source.isnumeric():
                        cv2.imshow("show fish", im0) 
                        cv2.waitKey(1)  # 1 millisecond
            # ------------------------------------------------------------------------------------------------------------------------- #
        else:
            msg_box_name = QMessageBox() 
            msg_box_name.setIcon(QMessageBox.Warning)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("img_fish/fish.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            msg_box_name.setWindowIcon(icon)
            msg_box_name.setText("please select fish !!!")
            msg_box_name.setWindowTitle("Warning")
            msg_box_name.setStandardButtons(QMessageBox.Ok)
            msg_box_name.exec_()

    def stop_work(self):
        import cv2
        self.start.show()
        checkboxs = [self.checkBox_hang_lueang, self.checkBox_khang_pan, 
                     self.checkBox_pod, self.checkBox_ku_lare,
                     self.checkBox_see_kun, self.checkBox_too, 
                     self.checkBox_sai_dang, self.checkBox_sai_dum,
                     self.checkBox_all]
        for checkbox in checkboxs:
            checkbox.setCheckState(False)
        cv2.destroyAllWindows()
        #raise StopIteration
        #convenyor_stop()


    def stop_exit(self):
        self.stop_work()
        self.exit.clicked.connect(exit)
        #print("exit")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())