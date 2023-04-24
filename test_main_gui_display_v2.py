import sys
import unittest
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QTimer
import time

import main_gui_display_v2

class TestMainFinalV2(unittest.TestCase):

    # Test hidden start button
    def test_hidden_start_button(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.checkBox_hang_lueang.setChecked(True)
        #QTest.mouseClick(self.form.start, Qt.LeftButton)
        self.form.start.click()

        self.assertTrue(self.form.start.isHidden())

    # when select all fish (checkbox), each fish checkbox is selected
    def test_select_all(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.checkBox_all.setCheckState(True)

        self.assertTrue(self.form.checkBox_pod.isChecked())
        self.assertTrue(self.form.checkBox_ku_lare.isChecked())
        self.assertTrue(self.form.checkBox_see_kun.isChecked())
        self.assertTrue(self.form.checkBox_too.isChecked())
        self.assertTrue(self.form.checkBox_khang_pan.isChecked())
        self.assertTrue(self.form.checkBox_hang_lueang.isChecked())
        self.assertTrue(self.form.checkBox_sai_dang.isChecked())
        self.assertTrue(self.form.checkBox_sai_dum.isChecked())

    def test_select_one(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        fish_checkbox = [self.form.checkBox_pod, self.form.checkBox_ku_lare, self.form.checkBox_see_kun,
                        self.form.checkBox_too, self.form.checkBox_khang_pan, self.form.checkBox_hang_lueang,
                        self.form.checkBox_sai_dang, self.form.checkBox_sai_dum]
        for checkbox in fish_checkbox:
            checkbox.setChecked(True)
            self.assertTrue(checkbox.isChecked())


    # when click stop button , reset checkbox and show start button
    def test_click_stop_reset_checkbox(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.stop.click()

        self.assertFalse(self.form.checkBox_pod.isChecked())
        self.assertFalse(self.form.checkBox_ku_lare.isChecked())
        self.assertFalse(self.form.checkBox_see_kun.isChecked())
        self.assertFalse(self.form.checkBox_too.isChecked())
        self.assertFalse(self.form.checkBox_khang_pan.isChecked())
        self.assertFalse(self.form.checkBox_hang_lueang.isChecked())
        self.assertFalse(self.form.checkBox_sai_dang.isChecked())
        self.assertFalse(self.form.checkBox_sai_dum.isChecked())
        self.assertFalse(self.form.start.isHidden())
        self.assertEqual(self.form.label.text(), "")

    def test_button_exit(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.exit.click()
        
        self.assertEqual(self.form.label.text(), "")

    # test value from sellect fish checkbox
    def test_checkbox_all(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.checkBox_all.setCheckState(True)
        self.form.start.click()

        self.assertEqual(self.form.selected_fish, ['pod', 'ku_lare', 'see_kun', 'too', 'khang_pan', 'hang_lueang', 'sai_dang', 'sai_dum'])

    def test_checkbox_one(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)
        fish_checkbox = [self.form.checkBox_pod, self.form.checkBox_ku_lare, self.form.checkBox_see_kun,
                        self.form.checkBox_too, self.form.checkBox_khang_pan, self.form.checkBox_hang_lueang,
                        self.form.checkBox_sai_dang, self.form.checkBox_sai_dum]
        name_fish = ['pod', 'ku_lare', 'see_kun', 'too', 'khang_pan', 'hang_lueang', 'sai_dang', 'sai_dum']
        for i in range(len(fish_checkbox)):
            fish_checkbox[i].setChecked(True)
            self.form.start.click()
            self.assertEqual(self.form.selected_fish, [name_fish[i]])
            self.form.stop.click()


    # when not select checkbox --> show pop up
    def test_show_pop_up(self):
        self.app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.form = main_gui_display_v2.Ui_MainWindow()
        self.form.setupUi(MainWindow)

        self.form.start.click()
        # show pop up
        self.assertEqual(self.form.msg_box_name.text(), "please select fish !!!")
        #self.form.msg_box_name.close()


if __name__ == '__main__':
    unittest.main()