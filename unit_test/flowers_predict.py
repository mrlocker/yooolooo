from yolo_model import YOLO_V3
import utils
def test_yolov3_predict():
    y = YOLO_V3(utils.load_json('configs/config_classification_flowers17.json'))
    while True:
        y.predict_classification(image_path="/Users/shidanlifuhetian/All/data/flowers17_tsycnh/test")

if __name__ == "__main__":
    test_yolov3_predict()