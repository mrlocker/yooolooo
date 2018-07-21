from yolo_model import prepare_data,YOLO_V3
from utils import load_json
if __name__ == "__main__":
    config = load_json('./config.json')
    train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'])
    a = YOLO_V3(config=config)
    a.train(train_gen, val_gen)
