import unittest
import utils

class UtilsTest(unittest.TestCase):
    def test_rect_interaction(self):
        rect1 = [0, 0, 10, 10]
        rect2 = [0, 0, 10, 10]
        area = utils.rect_interaction(rect1, rect2)
        self.assertEqual(area, 100)
    def test_iou(self):
        rect1 = [0, 0, 10, 10]
        rect2 = [0, 0, 10, 10]
        rect3 = [0,0,5,5]
        iou = utils.iou(rect1,rect2)
        iou2 = utils.iou(rect1,rect3)
        self.assertEqual(iou,1)
        self.assertEqual(0.25,iou2)
if __name__ == "__main__":
    unittest.main()



