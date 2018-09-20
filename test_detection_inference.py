from yolo_model import YOLO_V3,prepare_data
from keras.optimizers import Adam
import tensorflow as tf
from utils import load_json
from data_generator import St_Generator


if __name__ == "__main__":
    config = load_json('./configs/config_detection_defects_winK40.json')
    config['model']['debug'] = True
    config['model']['batch_size']=4
    gen = St_Generator(config,phase="train")
    val_gen = St_Generator(config,phase="test")

    yolo = YOLO_V3(config=config)
    yolo.load_weights(yolo.config['model']['final_model_weights'])  # TODO:临时注释，正式版需取消注释

    # result = yolo.model.predict_generator(val_gen)
    # yolo.inference(result)
    result = yolo.evaluate(val_gen)
    # print(result)