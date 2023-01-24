### capture camera by video ###

import uuid, cv2, time

cam = cv2.VideoCapture(0) # open camera 0 
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
start = time.time()

while True :
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    unique_file_name = str(uuid.uuid4())

    # auto save image 5 seconds
    
    
    stop = time.time() - start
    if stop >= 5:
        start = time.time()
        cv2.imwrite(f"{unique_file_name}.jpg", frame)
        print(unique_file_name)
    
    
    # manual save image
    if cv2.waitKey(1) & 0xFF == ord('s') :
        cv2.imwrite(f"{unique_file_name}.jpg", frame)
        print(unique_file_name)
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cam.release()
cv2.destroyAllWindows()