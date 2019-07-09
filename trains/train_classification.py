from yolo_model import prepare_data,YOLO_V3
from utils import load_json
import time,os
if __name__ == "__main__":

    config = load_json('configs/config_classification_flowers17_win2070.json')
    train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'],config['model']['batch_size'])
    yolo = YOLO_V3(config=config)
    yolo.backbone.trainable=False
    #
    # for layer in yolo.model.layers:
    #     trainable_point = False
    #     if layer.name == "dense_1":#"conv2d_52":  # 3x3/2 filters:1024
    #         trainable_point = True
    #     if trainable_point == True:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False
    format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_dir = "./logs/"+format_time
    os.mkdir(log_dir)
    yolo.train(train_gen, val_gen,log_dir)#  the model will be compiled before training
