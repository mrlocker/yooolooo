from yolo_model import prepare_data,YOLO_V3
from utils import load_json
from data_generator import St_Generator
from utils import get_dir_filelist_by_extension
if __name__ == "__main__":
    # this is for detection training
    config = load_json('./config_detection.json')
    # train_gen, val_gen = prepare_data(config['train']['data_folder'],config['val']['data_folder'])

    image_extention='bmp'
    img_list = get_dir_filelist_by_extension(dir=config['train']['data_folder']+'/images',ext=image_extention)
    img_list.sort()
    all_image_and_anno_paths = []
    for img_path in img_list:
        xmlname = img_path.split('/')[-1].replace(image_extention,'xml')
        anno_path = config['train']['data_folder']+'/annotations/'+xmlname
        all_image_and_anno_paths.append({
            'image_path':img_path,
            'anno_path':anno_path
        })
    gen = St_Generator(all_image_and_anno_paths,config)


    yolo = YOLO_V3(config=config)
    yolo.train_detection(gen,gen)
    # yolo.train(train_gen, val_gen)
