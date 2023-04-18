import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('PyQt5 App')
        
        self.capture = None
        
        button1 = QPushButton('Start Video', self)
        button1.move(50, 50)
        button1.clicked.connect(self.startVideoCapture)
        
        button2 = QPushButton('Stop Video', self)
        button2.move(150, 50)
        button2.clicked.connect(self.stopVideoCapture)
        
        self.show()
        
    def startVideoCapture(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print('Unable to open camera')
            return
        while True:
            ret, frame = self.capture.read()
            if ret:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()
        self.capture.release()
        
    def stopVideoCapture(self):
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())