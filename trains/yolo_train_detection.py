from yolo_model import YOLO_V3
from utils import load_json
from data_generator import St_Generator
if __name__ == "__main__":
    # this is for detection training
    config = load_json('configs/config_detection_defects_winK40.json')
    gen = St_Generator(config,phase="train",shuffle=True)
    val_gen = St_Generator(config,phase="test")

    yolo = YOLO_V3(config=config)

    trainable_point = False
    for layer in yolo.backbone.layers:
        if layer.name == "conv2d_44":# 3x3/2 filters:1024
            trainable_point = True
        if trainable_point == True:
            layer.trainable = True
        else:
            layer.trainable = False
    # contiune_training = False
    # if contiune_training:
    #     yolo.model.load_weights('')
    yolo.train_detection(gen,val_gen)
