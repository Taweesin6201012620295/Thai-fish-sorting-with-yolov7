from edit_bboxes import edit_boxes, check_range, in_range
import unittest


class TestBboxMethods(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # input test 
        self.array1 = [3,       0.729948, 0.218519, 0.116146, 0.437037, 0.41455 ]
        self.array2 = [2,       0.730729, 0.219444, 0.117708, 0.437037, 0.523577]
        self.array3 = [3,       0.84729, 0.509444, 0.12308, 0.257037, 0.823577]

    def test_main_input1(self):
        self.assertEqual(edit_boxes([]), [])
        # input 1 
        self.assertIn(self.array1, edit_boxes([self.array1]))
        self.assertEqual(edit_boxes([self.array1]), [self.array1])
        self.assertNotIn(self.array2, edit_boxes([self.array1]))
        self.assertNotEqual(edit_boxes([self.array1]), [])

    def test_main_input2(self):
        # input 2
        self.assertIn(self.array2, edit_boxes([self.array1, self.array2]))
        self.assertEqual(edit_boxes([self.array1, self.array2]), [self.array2])
        self.assertNotEqual(edit_boxes([self.array1, self.array2]), [self.array1])
        self.assertNotEqual(edit_boxes([self.array1, self.array2]), [self.array3])
        self.assertNotEqual(edit_boxes([self.array1, self.array2]), [])

    def test_main_input3(self):
        # input 3 
        self.assertEqual(edit_boxes([self.array1, self.array2, self.array3]), [self.array2, self.array3])
        self.assertIn(self.array2, edit_boxes([self.array1, self.array2, self.array3]))
        self.assertIn(self.array3, edit_boxes([self.array1, self.array2, self.array3]))
        self.assertIsNotNone(edit_boxes([self.array1, self.array2, self.array3]))
        self.assertNotIn(self.array1, edit_boxes([self.array1, self.array2, self.array3]))
        self.assertNotEqual(edit_boxes([self.array1, self.array2, self.array3]), [])
        self.assertNotEqual(edit_boxes([self.array1, self.array2, self.array3]), [self.array1])
        self.assertNotEqual(edit_boxes([self.array1, self.array2, self.array3]), [self.array2])
        self.assertNotEqual(edit_boxes([self.array1, self.array2, self.array3]), [self.array1, self.array3])


    def test_check_range(self):
        # check 2 input function
        self.assertIsNotNone(check_range(self.array2, self.array1))
        self.assertEqual(check_range(self.array1, self.array2), self.array2)
        self.assertEqual(check_range(self.array2, self.array1), self.array2)
        self.assertNotEqual(check_range(self.array1, self.array2), self.array1)
        self.assertNotEqual(check_range(self.array1, self.array2), False)
        self.assertNotEqual(check_range(self.array1, self.array2), [])

    def test_in_range(self):
        # check range 2 array
        x1 = self.array1[1]
        x2 = self.array2[1]
        x3 = self.array3[1]
        y1 = self.array1[2]
        y2 = self.array2[2]
        y3 = self.array3[2]

        self.assertIsNotNone(in_range(x1, x2))
        self.assertIsNotNone(in_range(x1, x3))
        self.assertIsNotNone(in_range(x2, x3))

        self.assertEqual(in_range(x1, x2), True)
        self.assertEqual(in_range(y1, y2), True)
        self.assertEqual(in_range(x1, x3), False)
        self.assertEqual(in_range(y1, y3), False)




if __name__ == "__main__":
    unittest.main()
    