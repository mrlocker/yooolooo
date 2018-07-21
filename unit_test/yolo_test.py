from yolo_model import YOLO_V3
import utils
def test_yolov3_predict():

    y = YOLO_V3(utils.load_json('./config.json'))
    y.construct_model()
    y.load_weights('./fl_model.h5')
    while True:
        y.predict(image_path="./flowers17/train")

if __name__ == "__main__":
    test_yolov3_predict()