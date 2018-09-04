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
    # phase could be "train" or "test" or "val", both test and val refer to val in config
    def __init__(self,config,shuffle=False,phase="train"):

        image_extention = 'bmp'
        if phase == "train":
            img_list = get_dir_filelist_by_extension(dir=config['train']['data_folder'] + '/images', ext=image_extention)
        elif phase == "test"or"val":
            img_list = get_dir_filelist_by_extension(dir=config['val']['data_folder'] + '/images', ext=image_extention)
        else:
            raise Exception("wrong phase! should be train test or val")
        img_list.sort()
        all_image_and_anno_paths = []
        for img_path in img_list:
            xmlname = img_path.split('/')[-1].replace(image_extention, 'xml')
            if phase == "train":
                anno_path = config['train']['data_folder'] + '/annotations/' + xmlname
            elif phase == "test"or"val":
                anno_path = config['val']['data_folder'] + '/annotations/' + xmlname

            all_image_and_anno_paths.append({
                'image_path': img_path,
                'anno_path': anno_path
            })



        self.image_anno_list = all_image_and_anno_paths
        self.batch_size = config['model']['batch_size']
        self.image_size = config['model']['image_size']
        self.grid       = config['model']['grid']
        self.classes    = config['model']['classes']
        self.anchors    = config['model']['anchors']
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_data()
        # image augmentation
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_pipe = iaa.Sequential([SquarePad(),
                                        sometimes(iaa.Fliplr(p=1)),
                                        sometimes(iaa.Flipud(p=1)),
                                        iaa.Scale({"height": self.image_size[0], "width": self.image_size[1]})])

    def __len__(self):
        return int(len(self.image_anno_list)/self.batch_size)
    def __getitem__(self, idx):
        x_batch = np.zeros(shape=(self.batch_size,self.image_size[0],self.image_size[1],3))
        y_batch = np.zeros(shape=(self.batch_size,self.grid,self.grid,3,4+1+len(self.classes)))#8*13*13*3*(4+1+classes) batch size*grid width*grid height*anchors*(4+1+classes)
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

        for i,img in enumerate(self.aug_imgs):
            x_batch[i] = img
        y_batches = self.create_y_true(self.aug_bbses,self.aug_labels,self.anchors)
        return x_batch,y_batches #ybatches order in Large anchor,medium anchor,small anchor

    def on_epoch_end(self):# modify dataset at each end of the epoch
        if self.shuffle:
            self.shuffle_data()
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

    def create_y_true(self,aug_bbses,aug_labels,anchors):
        # 3 y_batch,for 3 scale.First for large anchors, second for medium anchor, last for small anchors
        # grids are 13*13 26*26 52*52
        y_batches = [np.zeros(shape=(self.batch_size,self.grid*(2**l),self.grid*(2**l),3,4+1+len(self.classes))) for l in range(3)]#8*13*13*3*(4+1+classes) batch size*grid width*grid height*anchors*(4+1+classes)

        for i in range(len(aug_bbses)):#循环每一张图像对应的bboxes,每次一个batch中的一张，i即index of batch
            for j in range(len(aug_bbses[i].bounding_boxes)):#循环单张图像上所有的bbox
                rice = np.zeros(4+1+len(self.classes))#一个anchor中的内容
                rice[4] = 1# set confidence
                label_index = self.classes.index(aug_labels[i][j])
                rice[5+label_index]=1   # set one-hot label

                max_iou = 0
                anchor_index = -1# should be 0,1,2, 3,4,5, 6,7,8
                bbox = aug_bbses[i].bounding_boxes[j]

                for k in range(0,len(anchors),2):
                    rect1 = [0,0,bbox.width,bbox.height]
                    rect2 = [0,0,anchors[k],anchors[k+1]]
                    current_iou = utils.iou(rect1,rect2)
                    if current_iou>max_iou:
                        anchor_index = k/2
                        max_iou = current_iou
                y_batch_index = int(anchor_index//3)
                if y_batch_index == 0:
                    y_batch_index = 2
                elif y_batch_index == 2:
                    y_batch_index = 0
                inner_index = int(anchor_index%3)

                # different scales have different cell sizes
                cell_size = self.image_size[0] / (self.grid*(2**y_batch_index))

                # calc cell index start from 0
                cx = int(np.floor(bbox.center_x/cell_size))
                cy = int(np.floor(bbox.center_y/cell_size))
                x_new = (bbox.center_x-cx*cell_size)/cell_size  # scale to 0~1 (relative to cell size)
                y_new = (bbox.center_y-cy*cell_size)/cell_size
                w_new = bbox.width/cell_size                    # scale to 0~13 (grid size)
                h_new = bbox.height/cell_size
                rice[0:4]=[x_new,y_new,w_new,h_new]


                y_batches[y_batch_index][i][cy][cx][inner_index] = rice  # TODO:cx cy or cy cx? seems cy cx is right
                # check_grid(self.aug_imgs[i],self.aug_bbses[i],self.aug_labels[i],cx, cy,self.grid*(2**y_batch_index))
        return y_batches
    def choose_anchor(self):
        pass
    def shuffle_data(self):
        random.seed(time.time())
        random.shuffle(self.image_anno_list)
def check_grid(img,bbs,labels,cx,cy,grid):
    img = img.copy()
    w,h,_ = img.shape
    cellsize = int(w/grid)
    x1 = int(cx*cellsize)
    y1 = int(cy*cellsize)
    x2 = int((cx+1)*cellsize)
    y2 = int((cy+1)*cellsize)
    # roi = img[x1:x2,y1:y2]#wh
    roi = img[y1:y2,x1:x2] # hw
    img = draw_aug_bboxes(img,bbs,labels)
    for i in range(grid):
        img = cv2.line(img,(i*cellsize,0),(i*cellsize,h-1),color=(0,255,0))
        img = cv2.line(img,(0,i*cellsize),(h-1,i*cellsize),color=(0,255,0))
    cv2.imshow('aug img',img)
    cv2.imshow('cell',roi)
    cv2.waitKey(0)
def draw_aug_bboxes(aug_img, bboxes,labels):
    pred_color = (255,255,255)
    for i,rect in enumerate(bboxes.bounding_boxes):
        cv2.rectangle(aug_img, pt1=(rect.x1, rect.y1), pt2=(rect.x2, rect.y2), color=pred_color, thickness=2)
        cv2.putText(aug_img, labels[i] + ' ', (rect.x1 + 2, rect.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color)
        cv2.circle(aug_img,center=(int(rect.center_x),int(rect.center_y)),radius=1,color=(0,0,255),thickness=-1)

    return aug_img

if __name__ == "__main__":
    config = load_json('./config_detection.json')
    # image_extention='bmp'
    # img_list = get_dir_filelist_by_extension(dir=config['train']['data_folder']+'/images',ext=image_extention)
    # img_list.sort()
    # all_image_and_anno_paths = []
    # for img_path in img_list:
    #     xmlname = img_path.split('/')[-1].replace(image_extention,'xml')
    #     anno_path = config['train']['data_folder']+'/annotations/'+xmlname
    #     all_image_and_anno_paths.append({
    #         'image_path':img_path,
    #         'anno_path':anno_path
    #     })
    gen = St_Generator(config)
    print('len:',len(gen))
    one_batch = gen.__getitem__(0)
    for i in range(5):
        img = gen.aug_imgs[i]
        img_annos = gen.aug_bbses[i]
        img_labels = gen.aug_labels[i]
        after_img = draw_aug_bboxes(img,img_annos,img_labels)
        cv2.imshow('img',after_img)
        cv2.waitKey(0)