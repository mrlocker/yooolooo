from yolo_model import YOLO_V3
from utils import load_json
from data_generator import St_Generator
if __name__ == "__main__":
    # this is for detection training
    config = load_json('./configs/config_detection_defects_win.json')
    gen = St_Generator(config,phase="train")
    val_gen = St_Generator(config,phase="test")

    yolo = YOLO_V3(config=config)

    yolo.evaluate(val_gen)