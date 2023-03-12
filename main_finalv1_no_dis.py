from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

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
        self.start.clicked.connect(self.select)
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


    def check_select(self):
        store_select = []
        self.state_fish = False
        if self.checkBox_pod.isChecked():
            store_select.append(0)
            self.state_fish = True
        if self.checkBox_ku_lare.isChecked():
            store_select.append(1)
            self.state_fish = True
        if self.checkBox_see_kun.isChecked():
            store_select.append(2)
            self.state_fish = True
        if self.checkBox_too.isChecked():
            store_select.append(3)
            self.state_fish = True
        if self.checkBox_khang_pan.isChecked():
            store_select.append(4)
            self.state_fish = True
        if self.checkBox_hang_lueang.isChecked():
            store_select.append(5)            
            self.state_fish = True
        if self.checkBox_sai_dang.isChecked():
            store_select.append(6)
            self.state_fish = True
        if self.checkBox_sai_dum.isChecked():
            store_select.append(7)
        return store_select
        
    def select(self):
        print("start")
        self.check_select()
        if self.state_fish:
            self.start.hide()
        else:
            msg_box_name = QMessageBox() 
            msg_box_name.setIcon(QMessageBox.Warning)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("img_fish/fish.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            msg_box_name.setWindowIcon(icon)
            msg_box_name.setText("please select fish""")
            msg_box_name.setWindowTitle("Warning")
            msg_box_name.setStandardButtons(QMessageBox.Ok)
            msg_box_name.exec_()

    def stop_work(self):
        self.start.show()
        checkboxs = [self.checkBox_hang_lueang, self.checkBox_khang_pan, 
                     self.checkBox_pod, self.checkBox_ku_lare,
                     self.checkBox_see_kun, self.checkBox_too, 
                     self.checkBox_sai_dang, self.checkBox_sai_dum,
                     self.checkBox_all]
        for checkbox in checkboxs:
            checkbox.setCheckState(False)


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
