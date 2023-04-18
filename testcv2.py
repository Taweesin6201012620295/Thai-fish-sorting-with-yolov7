# import the opencv library
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
print("Camera Opened: ", vid.isOpened())

while(True):
    # Capture the video frame
    ret, frame = vid.read()
    if ret:
        cv2.imshow('Test Window', frame)
    else:
        print("Error Drawing Frame")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()