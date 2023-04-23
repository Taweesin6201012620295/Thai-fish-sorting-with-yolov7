from main_gui_display_v2 import *
import unittest

best_case = ["too","too","too","too"]                   #TEST OK too
normal_case1 = ["too","ku_lare","too","sai_dang"]       #TEST OK too
normal_case2 = ["too","ku_lare","too","ku_lare"]        #TEST OK too
worst_case = ["too","ku_lare","sai_dang","sai_dum"]   #TEST OK too
sad_path = ["too", "ku_lare", "ku_lare","sai_dang"]     #TEST OK ku_lare
sad_path2 = [ ] #TEST None

class_window = Ui_MainWindow()

class TestArray(unittest.TestCase):
    def test_best_case(self):
        self.assertEqual(class_window.max_array(best_case),"too")

    def test_normal_case1(self):
        self.assertEqual(class_window.max_array(normal_case1),"too")

    def test_normal_case2(self):
        self.assertEqual(class_window.max_array(normal_case2),"too")

    def test_worst_case(self):
        self.assertEqual(class_window.max_array(worst_case),"too")

    def test_sad_path(self):
        self.assertNotEqual(class_window.max_array(sad_path),"too")

    def test_sad_path2(self):
        self.assertIsNone(class_window.max_array(sad_path2))

if __name__ == '__main__':
   unittest.main()