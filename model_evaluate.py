from yolo_model import YOLO_V3,prepare_data
from keras.optimizers import Adam
import tensorflow as tf
from utils import load_json


if __name__ == "__main__":
    config = load_json('./config.json')
    train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'])
    yolo = YOLO_V3(config=config)

    yolo.load_weights('tmp/fl_ckpt.h5')

    # result = yolo.model.predict_generator(val_gen)
    result = yolo.evaluate(val_gen)
    print(result)