from PyQt5 import QtWidgets
from main_finalv2 import Ui_MainWindow
import pytest
import sys


def test_checkBox_all():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # checkbox 
    ui.checkBox_all.setChecked(True)
    assert ui.checkBox_hang_lueang.isChecked
    assert ui.checkBox_khang_pan.isChecked
    assert ui.checkBox_pod.isChecked
    assert ui.checkBox_ku_lare.isChecked
    assert ui.checkBox_sai_dang.isChecked
    assert ui.checkBox_sai_dum.isChecked
    assert ui.checkBox_too.isChecked
    assert ui.checkBox_see_kun.isChecked

def test_checkBox_one():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # checkbox hang_lueang
    ui.checkBox_hang_lueang.setChecked(True)
    assert ui.checkBox_hang_lueang.isChecked
    # checkbox khang_pan
    ui.checkBox_khang_pan.setChecked(True)
    assert ui.checkBox_khang_pan.isChecked
    # checkbox pod
    ui.checkBox_pod.setChecked(True)
    assert ui.checkBox_pod.isChecked
    # checkbox ku_lare
    ui.checkBox_ku_lare.setChecked(True)
    assert ui.checkBox_ku_lare.isChecked
    # checkbox too
    ui.checkBox_too.setChecked(True)
    assert ui.checkBox_too.isChecked
    # checkbox see_kun
    ui.checkBox_see_kun.setChecked(True)
    assert ui.checkBox_see_kun.isChecked
    # checkbox sai_dang
    ui.checkBox_sai_dang.setChecked(True)
    assert ui.checkBox_sai_dang.isChecked
    # checkbox sai_dum
    ui.checkBox_sai_dum.setChecked(True)
    assert ui.checkBox_sai_dum.isChecked

def test_button_start_with_checkBox():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # checkbox 
    ui.checkBox_all.setChecked(True)
    # click start button
    ui.start.click()
    assert ui.start.isHidden
    ui.stop.click()
    ui.exit.click()


def test_button_start_without_checkBox():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # click start button
    ui.start.click()


def test_button_stop():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # click stop button
    ui.stop.click()
    assert ui.label.text() == ""


def test_button_exit():
    # open window
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # click stop button
    ui.exit.click()
    assert ui.label.text() == ""
    