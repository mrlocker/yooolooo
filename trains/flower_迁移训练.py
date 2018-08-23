from yolo_model import prepare_data,YOLO_V3
from utils import load_json
if __name__ == "__main__":

    config = load_json('./config.json')
    train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'])
    yolo = YOLO_V3(config=config)
    for layer in yolo.backbone.layers:
        layer.trainable = False
    contiune_training = True
    if contiune_training:
        yolo.model.load_weights('./tmp/fl_ckpt.h5')
    yolo.train(train_gen, val_gen)#  the model will be compiled before training
