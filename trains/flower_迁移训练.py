from yolo_model import prepare_data,YOLO_V3
from utils import load_json
if __name__ == "__main__":

    config = load_json('configs/config_classification_flowers17_win.json')
    train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'],config['model']['batch_size'])
    yolo = YOLO_V3(config=config)
    for layer in yolo.backbone.layers:
        layer.trainable = True
    contiune_training = True
    if contiune_training:
        yolo.model.load_weights('tmp/classification_flowers_ckpt_tl.h5')
    yolo.train(train_gen, val_gen)#  the model will be compiled before training
