from keras.utils import Sequence
from keras.preprocessing.image import load_img

from utils import load_json,get_dir_filelist_by_extension
import random,time
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import utils
import cv2
import xml.etree.ElementTree as ET

class SquarePad(iaa.Augmenter):

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(SquarePad, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.all_offsets = []
        self.all_new_shapes=[]

    def func_images(self,images, random_state, parents, hooks):
        # print('进入func_images')
        # print('图片数量：', len(images))
        self.all_offsets = []
        self.all_new_shapes = []
        for i, image in enumerate(images):
            # print('image index: ',i)
            img_rows, img_cols, img_channels = image.shape
            bg_size = 0
            if img_rows < img_cols:  # 情况1
                bg_size = img_cols
                max_offset_y = bg_size - img_rows
                offset_y = random.randint(0, max_offset_y)
                offset_x = 0
            elif img_rows > img_cols:  # 情况2
                bg_size = img_rows
                max_offset_x = bg_size - img_cols
                offset_x = random.randint(0, max_offset_x)
                offset_y = 0
            else:
                bg_size = img_rows
                offset_x = offset_y = 0
            bg = np.zeros([bg_size, bg_size, img_channels], dtype=image.dtype)
            bg = self.paste(bg, image, offset_x, offset_y)
            images[i] = bg
            self.all_offsets.append({
                'offset_x': offset_x,
                'offset_y': offset_y
            })
            self.all_new_shapes.append((bg_size,bg_size,img_channels))
        return images

    def func_keypoints(self,keypoints_on_images, random_state, parents, hooks):
        # print('进入func_keypoints')
        # print(keypoints_on_images)
        for i in range(len(keypoints_on_images)):
            for j in range(len(keypoints_on_images[i].keypoints)):
                keypoints_on_images[i].keypoints[j].x += self.all_offsets[i]['offset_x']
                keypoints_on_images[i].keypoints[j].y += self.all_offsets[i]['offset_y']
                keypoints_on_images[i].shape = self.all_new_shapes[i]
        return keypoints_on_images
    def _augment_images(self, images, random_state, parents, hooks):
        return self.func_images(images, random_state, parents=parents, hooks=hooks)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = self.func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks)
        assert isinstance(result, list)
        assert all([isinstance(el, ia.KeypointsOnImage) for el in result])
        return result

    def get_parameters(self):
        return []

    def paste(self,background, image, offset_x, offset_y):
        background[offset_y:image.shape[0] + offset_y,  # 竖直方向偏移
        offset_x:image.shape[1] + offset_x  # 水平方向偏移
        ] = image
        return background

class St_Generator(Sequence):
    def __init__(self,image_anno_list,config,shuffle=False):
        self.image_anno_list = image_anno_list
        self.batch_size = config['model']['batch_size']
        self.image_size = config['model']['image_size']
        self.grid       = config['model']['grid']
        self.classes    = config['model']['classes']
        if shuffle:
            random.seed(time.time())
            random.shuffle(image_anno_list)
        # image augmentation
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_pipe = iaa.Sequential([SquarePad(),
                                        sometimes(iaa.Fliplr(p=1)),
                                        sometimes(iaa.Flipud(p=1)),
                                        iaa.Scale({"height": self.image_size[0], "width": self.image_size[1]})])

    def __getitem__(self, idx):
        x_batch = np.zeros(shape=(self.batch_size,self.image_size[0],self.image_size[1],3))
        y_batch = np.zeros(shape=(self.batch_size,self.grid,self.grid,3,4+1+len(self.classes)))#8*13*13*3*8 batch size*grid width*grid height*anchors*classes
        images = []
        images_bboxes = []
        images_labels = []
        for i in range(self.batch_size):
            img = load_img(self.image_anno_list[idx+i]['image_path'])
            img = np.array(img)
            images.append(img)

            annos = self.parse_annotation(path=self.image_anno_list[idx+i]['anno_path'])
            bboxes,labels = self.annos_to_bbox(annos)
            images_bboxes.append(bboxes)
            images_labels.append(labels)
        aug_pipe_det = self.aug_pipe.to_deterministic()
        self.aug_imgs = aug_pipe_det.augment_images(images)
        self.aug_bbses = aug_pipe_det.augment_bounding_boxes(images_bboxes)
        self.aug_labels = images_labels
        # 至此，扩增完毕
        pass

            # x_batch[i] = img

    def __len__(self):
        pass
    def on_epoch_end(self):# modify dataset at each end of the epoch
        pass
    # def aug_image(self,l_bound,r_bound):
    #     # 2.读取图像和label(每次一批）
    #     bbses = []
    #     imgs = []
    #     ys = []
    #     for i in range(l_bound, r_bound):
    #         img = cv.imread(self.images_with_objs[i]['filename'])
    #         annotation = self.images_with_objs[i]['object']
    #         # --------------
    #         tmp_bbox = []
    #         tmp_y = []
    #         for bbox in annotation:
    #             tmp_bbox.append(ia.BoundingBox(
    #                 x1=bbox['xmin']-1,
    #                 y1=bbox['ymin']-1,
    #                 x2=bbox['xmax']-1,
    #                 y2=bbox['ymax']-1,
    #             ))
    #             tmp_y.append(bbox['name'])
    #         bbs = ia.BoundingBoxesOnImage(tmp_bbox, shape=img.shape)
    #         # --------------
    #         imgs.append(img)
    #         bbses.append(bbs)
    #         ys.append(tmp_y)
    #
    #     #print('bbses before:',bbses)
    #     # 3. 图像预处理
    #     aug_pipe_det = self.aug_pipe.to_deterministic()
    #
    #     aug_imgs = aug_pipe_det.augment_images(imgs)
    #     aug_bbses = aug_pipe_det.augment_bounding_boxes(bbses)
    #     aug_clses = ys
    #
    #     #print('bbses after',)
    #     if self.debug:
    #         for i, _ in enumerate(aug_imgs):
    #             image_before = bbses[i].draw_on_image(imgs[i])
    #             image_after = aug_bbses[i].draw_on_image(aug_imgs[i])
    #             cv.imshow('before' + str(i + 1), image_before)
    #             cv.imshow('after' + str(i + 1), image_after)
    #         # pass
    #         # cv.waitKey(0)
    #     return aug_imgs,aug_bbses,aug_clses

    def parse_annotation(self,path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        annos = []

        size_node = tree.find('size')
        width = size_node.find('width').text
        height = size_node.find('height').text

        for elem in tree.iter('object'):
            name_node = elem.find('name')
            class_name = name_node.text

            bbox_node = elem.find('bndbox')
            xmin = bbox_node.find('xmin').text
            ymin = bbox_node.find('ymin').text
            xmax = bbox_node.find('xmax').text
            ymax = bbox_node.find('ymax').text



            annos.append({
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'label': class_name,
                'img_shape': [int(width), int(height)]
            })

        return annos

    def annos_to_bbox(self,annos):
        tmp_bbox = []
        labels = []
        for anno in annos:
            tmp_bbox.append(ia.BoundingBox(
                x1=anno['xmin'] - 1,  # -1 是因为pascal voc的标记格式问题
                y1=anno['ymin'] - 1,
                x2=anno['xmax'] - 1,
                y2=anno['ymax'] - 1,
            ))
            labels.append(anno['label'])
        bboxes = ia.BoundingBoxesOnImage(tmp_bbox, shape=annos[0]['img_shape'])
        return bboxes, labels

def draw_aug_bboxes(aug_img, bboxes,labels):
    pred_color = (255,255,255)
    for i,rect in enumerate(bboxes.bounding_boxes):
        cv2.rectangle(aug_img, pt1=(rect.x1, rect.y1), pt2=(rect.x2, rect.y2), color=pred_color, thickness=2)
        cv2.putText(aug_img, labels[i] + ' ', (rect.x1 + 2, rect.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color)

    return aug_img

if __name__ == "__main__":
    config = load_json('./miniset/test_config3.json')
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
    gen.__getitem__(0)
    for i in range(5):
        img = gen.aug_imgs[i]
        img_annos = gen.aug_bbses[i]
        img_labels = gen.aug_labels[i]
        after_img = draw_aug_bboxes(img,img_annos,img_labels)
        cv2.imshow('img',after_img)
        cv2.waitKey(0)