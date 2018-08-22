from yolo_model import YOLO_V3
import utils

def test_yolov3_classification():

    y = YOLO_V3(utils.load_json('./unit_test/test_config1.json'))


if __name__ == "__main__":
    test_yolov3_classification()