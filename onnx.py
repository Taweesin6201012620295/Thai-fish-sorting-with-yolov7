import cv2
onnx_path = "yolov7_3200.onnx"
net = cv2.dnn.readNetFromONNX(onnx_path)