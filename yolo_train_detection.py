from yolo_model import YOLO_V3
from utils import load_json
from data_generator import St_Generator
if __name__ == "__main__":
    # this is for detection training
    config = load_json('./config_detection.json')
    gen = St_Generator(config)

    yolo = YOLO_V3(config=config)
    yolo.train_detection(gen,gen)
