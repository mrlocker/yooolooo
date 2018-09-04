import keras
from keras.layers import Conv2D,Input,BatchNormalization,\
    LeakyReLU,Add,GlobalAveragePooling2D,Dense,Activation,\
    UpSampling2D,Concatenate,Reshape
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.losses import categorical_crossentropy,mean_squared_error
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard,ReduceLROnPlateau

import tensorflow as tf
import numpy as np
from utils import *
from keras import backend as K
import time

class Basic_Conv():
    def __init__(self,filters,kernel_size,strides=(1,1),batch_normalize=True,
                 pad='same',activation='leaky'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_normalize = batch_normalize
        self.pad = pad
        self.activation = activation
    def __call__(self, food):
        shit = Conv2D(filters=self.filters,
                      kernel_size=self.kernel_size,
                      strides=self.strides,padding=self.pad,use_bias=False)(food)
        shit = BatchNormalization()(shit)# 这里将BN放在了激活函数之前。和原作者代码保持一致
        if self.activation == 'leaky':
            shit = LeakyReLU(alpha=0.1)(shit)
        elif self.activation == 'linear':
            shit = Activation(self.activation)(shit)
        # shit = BatchNormalization()(shit)# 这里将BN放在了激活函数之后。

        return shit
class Basic_Res():# same to [shortcut] in yolov3.cfg
    def __init__(self,activation='linear'):
        self.activation = activation
    def __call__(self, shitA,shitB):
        shit = Add()([shitA,shitB])
        if self.activation == "linear":
            shit = Activation(self.activation)(shit)
        return shit
class Basic_Route():# same to [route] in yolov3.cfg
    def __call__(self, shitA,shitB=None):
        if shitB == None:
            return shitA
        if shitB != None:
            shit = Concatenate()([shitA, shitB])
            return shit
class Basic_Detection():
    def __call__(self, shit):
        shit = Reshape((shit.shape.dims[1].value, shit.shape.dims[2].value, 3, int(shit.shape.dims[3].value / 3)))(shit)
        return shit
class Bbox():
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.label = "Undefined"
        self.confidence = 0.0
class YOLO_V3():
    def __init__(self,config):
        self.config = config
        self.shits = []
        self.inputs = Input(shape=(self.config['model']['image_size'][0], self.config['model']['image_size'][1], 3))
        self.num_classes = len(self.config['model']['classes'])
        self.batch_size = self.config['model']['batch_size']
        self.debug = self.config['model']['debug']
        # yolo算法采用前后端分离。后端指的是主干网络。主干网络配合不同的前端，可以实现分类或者检测的目的。
        self.construct_backbone(self.inputs)
        # 载入预训练模型参数（仅主干网络）
        if self.config['model']['type'] == "classification":
            self.construct_classification_model()
        elif self.config['model']['type'] == "detection":
            self.construct_detection_model()
        #self.load_pretrain_weights()

        # official_backbone = load_model('weights/darknet53.h5',compile=False)
        # official_backbone.summary(positions=[.33, .6, .7, 1])
        # a=0
        # exit()

    def construct_backbone(self,inputs):
        # 该主干网络和yolo v3论文上花的那个图一模一样。不包括最后三层，那三层放到了前端里面
        self.shits.append(Basic_Conv(filters=32, kernel_size=3)(inputs))

        self.shits.append(Basic_Conv(filters=64, kernel_size=3, strides=(2, 2))(self.shits[-1]))

        self.shits.append(Basic_Conv(filters=32, kernel_size=1)(self.shits[-1]))
        self.shits.append(Basic_Conv(filters=64, kernel_size=3)(self.shits[-1]))
        self.shits.append(Basic_Res()(self.shits[-1], self.shits[-3]))

        self.shits.append(Basic_Conv(filters=128, kernel_size=3, strides=2)(self.shits[-1]))

        for i in range(2):
            self.shits.append(Basic_Conv(filters=64, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=128, kernel_size=3)(self.shits[-1]))
            self.shits.append(Basic_Res()(self.shits[-1], self.shits[-3]))

        # yolov3.cfg 113~283
        self.shits.append(Basic_Conv(filters=256, kernel_size=3, strides=2)(self.shits[-1]))
        for i in range(8):
            self.shits.append(Basic_Conv(filters=128, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=256, kernel_size=3)(self.shits[-1]))
            self.shits.append(Basic_Res()(self.shits[-1], self.shits[-3]))
        # yolov3.cfg 284~458
        self.shits.append(Basic_Conv(filters=512, kernel_size=3, strides=2)(self.shits[-1]))
        for i in range(8):
            self.shits.append(Basic_Conv(filters=256, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=512, kernel_size=3)(self.shits[-1]))
            self.shits.append(Basic_Res()(self.shits[-1], self.shits[-3]))
        # yolov3.cfg 459~547
        self.shits.append(Basic_Conv(filters=1024, kernel_size=3, strides=2)(self.shits[-1]))
        for i in range(4):
            self.shits.append(Basic_Conv(filters=512, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=1024, kernel_size=3)(self.shits[-1]))
            self.shits.append(Basic_Res()(self.shits[-1], self.shits[-3]))
        #
        self.backbone = Model(inputs,self.shits[-1])
    def construct_classification_model(self):
        self.shits.append(GlobalAveragePooling2D()(self.shits[-1]))
        output_units = self.config['model']['output_units']
        logits = Dense(units=output_units)(self.shits[-1])
        self.model = Model(inputs=self.inputs,outputs=logits)
        self.model.summary()

        plot_model(self.model)
        print("分类网络组建完毕")
    def construct_detection_model(self):

        for i in range(3):
            self.shits.append(Basic_Conv(filters=512,kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=1024,kernel_size=3)(self.shits[-1]))
        self.shits.append(Basic_Conv(filters=3*(4+1+self.num_classes),kernel_size=1,activation='linear')(self.shits[-1]))
        self.shits.append(Basic_Detection()(self.shits[-1]))#82 First Detection layer, anchors should be large(3 anchors)
        ##################################################
        self.shits.append(Basic_Route()(self.shits[-4]))
        self.shits.append(Basic_Conv(filters=256,kernel_size=1)(self.shits[-1]))
        self.shits.append(UpSampling2D()(self.shits[-1]))
        self.shits.append(Basic_Route()(self.shits[-1],self.shits[61]))
        for i in range(3):
            self.shits.append(Basic_Conv(filters=256, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=512, kernel_size=3)(self.shits[-1]))
        self.shits.append(Basic_Conv(filters=3*(4+1+self.num_classes),kernel_size=1,activation='linear')(self.shits[-1]))
        self.shits.append(Basic_Detection()(self.shits[-1]))#94 Second Detection layer, anchors should be medium(3 anchors)
        ##################################################
        self.shits.append(Basic_Route()(self.shits[-4]))
        self.shits.append(Basic_Conv(filters=128,kernel_size=1)(self.shits[-1]))
        self.shits.append(UpSampling2D()(self.shits[-1]))#out 52*52*128
        self.shits.append(Basic_Route()(self.shits[-1],self.shits[36]))
        for i in range(3):
            self.shits.append(Basic_Conv(filters=128, kernel_size=1)(self.shits[-1]))
            self.shits.append(Basic_Conv(filters=256, kernel_size=3)(self.shits[-1]))
        self.shits.append(Basic_Conv(filters=3*(4+1+self.num_classes),kernel_size=1,activation='linear')(self.shits[-1]))
        self.shits.append(Basic_Detection()(self.shits[-1]))#106 Third Detection layer, anchors should be small(3 anchors)
        ##################################################
        self.model = Model(inputs=self.inputs,outputs=[self.shits[82],self.shits[94],self.shits[106]])

        self.model.summary()
        plot_model(self.model)
        print("目标检测网络构建完毕")

    def train(self,train_generator,val_generator):
        config = self.config
        loss_func = tf.losses.softmax_cross_entropy
        filepath = "./tmp/classification_flowers_ckpt_{epoch:02d}_{val_acc:.2f}.h5"

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                     verbose=1, save_best_only=False)

        def lr_sch(epoch):
            # 200 total
            if epoch < 50:
                return 1e-3
            if 50 <= epoch < 100:
                return 1e-4
            if epoch >= 100:
                return 1e-5

        lr_scheduler = LearningRateScheduler(lr_sch)
        lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                       mode='max', min_lr=1e-3)
        tb = TensorBoard(log_dir='./logs',write_graph=False)
        callbacks = [checkpoint, lr_scheduler, lr_reducer,tb]
        self.model.compile(optimizer=Adam(),loss=loss_func,metrics=['accuracy'])
        self.model.fit_generator(generator=train_generator,
                                 epochs=config['train']['epochs'],
                                 validation_data=val_generator,
                                 validation_steps=280/self.batch_size,class_weight='auto',
                                 callbacks=callbacks)
        self.model.save_weights('fl_model.h5')
    def train_detection(self,train_generator,val_generator):
        filepath = "./tmp/detection_ckpt_{epoch:02d}_{val_acc:.2f}.h5"

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                     verbose=1, save_best_only=False)
        self.model.compile(optimizer=Adam(),loss=[self.yolo_loss,self.yolo_loss,self.yolo_loss])
        self.model.fit_generator(generator=train_generator,epochs=self.config['train']['epochs'],callbacks=[checkpoint])
    def load_pretrain_weights(self):
        # 加载预训练参数。首先加载完全模型的参数，如果没有再加载主干网络的参数。
        if self.config['model']['pretrain_full'] != "":
            self.model.load_weights(self.config['model']['pretrain_full'])
            print("全模型参数已加载：%s"%self.config['model']['pretrain_full'])
        elif self.config['model']['pretrain_backbone'] != "":
            self.backbone.load_weights(self.config['model']['pretrain_backbone'],by_name=True)
            print("backbone模型参数已加载：%s"%self.config['model']['pretrain_backbone'])
        else:
            print('！！未加载预训练参数')

    def yolo_loss(self,y_true, y_pred):
        # batch_index,cy,cx,sub_anchor_index,inner_index
        # 1. batch中的数据逐个循环
        print("y_true type:%s, y_pred type:%s" % (type(y_true), type(y_pred)))
        lambda_coord = 5
        lambda_noobj = 0.5

        # 1. prepare y_pred
        y_pred_xy = tf.sigmoid(y_pred[..., 0:2])  # scale x to 0~1
        y_pred_wh = tf.sigmoid(y_pred[..., 2:4])  # scale confidence to 0~1 #（4，13，13，3，1）
        y_pred_confidence = tf.sigmoid(y_pred[..., 4:5])
        y_pred_classes = y_pred[..., 5:]

        y_true_xy = y_true[..., 0:2]
        y_true_wh = y_true[..., 2:4]
        y_true_confidence = y_true[..., 4:5]
        y_true_classes = y_true[..., 5:]

        obj_mask = y_true[..., 4:5]  # 1 means the box exists object
        no_obj_mask = tf.subtract(1.0, obj_mask)

        # 2. calc xy loss
        xy_minus = tf.subtract(y_true_xy, y_pred_xy)
        xy_square = tf.square(xy_minus)
        xy_sum = tf.reduce_sum(xy_square, axis=-1, keep_dims=True)
        xy_loss = tf.reduce_sum(tf.multiply(xy_sum, obj_mask))
        # 3. calc wh loss
        wh_minus = tf.subtract(tf.sqrt(y_pred_wh), tf.sqrt(y_true_wh))
        wh_square = tf.square(wh_minus)
        wh_sum = tf.reduce_sum(wh_square, axis=-1, keep_dims=True)
        wh_loss = tf.reduce_sum(tf.multiply(wh_sum, obj_mask))
        # 4. calc confidence loss
        con_minus = tf.subtract(y_true_confidence, y_pred_confidence)
        con_square = tf.square(con_minus)
        con_loss = tf.reduce_sum(tf.multiply(con_square, obj_mask))
        no_con_loss = tf.reduce_sum(tf.multiply(con_square, no_obj_mask))
        # 5. calc classes loss
        y_pred_classes_soft = tf.nn.softmax(y_pred_classes, axis=-1)
        classes_minus = tf.subtract(y_true_classes, y_pred_classes_soft)
        classes_square = tf.square(classes_minus)
        classes_loss = tf.reduce_sum(tf.multiply(classes_square, obj_mask))

        # 6.total loss
        final_xy_loss = tf.multiply(xy_loss, tf.convert_to_tensor(lambda_coord, dtype=tf.float32))
        final_wh_loss = tf.multiply(wh_loss, tf.convert_to_tensor(lambda_coord, dtype=tf.float32))
        final_con_loss = con_loss
        final_no_con_loss = tf.multiply(no_con_loss, tf.convert_to_tensor(lambda_noobj, dtype=tf.float32))
        final_classes_loss = classes_loss

        total_loss = tf.add_n([final_xy_loss, final_wh_loss, final_con_loss, final_no_con_loss, final_classes_loss])
        # batch_size = y_pred.get_shape()[0].value
        total_loss = tf.divide(total_loss,self.batch_size)
        # total_loss = tf.Print(total_loss, [total_loss], message='total Loss \t')

        #########
        # init_op = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #
        #     print(sess.run([final_xy_loss,final_wh_loss,final_con_loss,final_no_con_loss,final_classes_loss]))
        #     print('total_loss :',sess.run(total_loss))
        #     tf.Print()

        return total_loss

    def load_weights(self,path):
        self.model.load_weights(path)
    def evaluate(self,generator):
        self.load_weights(self.config['model']['final_model_weights'])
        pr_result = self.model.predict_generator(generator)
        pass
    def predict_classification(self, image_path, threshold=0.5):
        # predict 用来预测单张图像的分类结果。
        import random,os,cv2
        true_class = random.randint(0,16)
        class_name = "flower_"+chr(65+true_class)
        target_folder = os.path.join(image_path,class_name)
        files = os.listdir(target_folder)
        fileindex = random.randint(0,len(files)-1)
        target_path = os.path.join(target_folder,files[fileindex])
        image = load_img(target_path,target_size=(256,256))

        np_image = np.array(image)
        image_for_cv2_show = np_image[:, :, ::-1].copy()

        image = np.expand_dims(np_image,axis=0)
        img = preprocess(image)
        result = self.model.predict(img)
        r = softmax(result)
        cv2.imshow('result', image_for_cv2_show)
        pred_index = np.argmax(r)
        pred = self.config['classes'][pred_index]
        # true_index = class_indices[true_class]# get one classes's index
        # index_classe = dict(zip(class_indices.values(), class_indices.keys()))

        if np.max(r) > 0.5:
            if pred == class_name:
                message = " 🍺"
            else:
                message = " 💀"
            print('真值种类名称：%s 预测类别名称：%s %s'%(class_name,pred,message))
        else:
            print('真值：',class_name,'预测值：无'," 💀")
        cv2.waitKey(0)

    def calc_classes_score(self,raw_output):
        confidence = raw_output[..., 4:5]
        classes = raw_output[..., 5:]
        classes_scores = classes * confidence
        # print(raw_output.shape,confidence.shape,classes.shape,classes_scores.shape)
        # print('classes:',classes[0,0,0,0])
        # print('confidence:',confidence[0,0,0,0])
        # print('cls_scores:',classes_scores[0,0,0,0])
        r = np.greater(classes_scores, self.config['model']['threshold'])
        # print('r:',r[0,0,0,0])
        r2 = np.where(r, classes_scores, 0)
        # print('r2:',r2[0,0,0,0])
        return r2

    def regulize_single_raw_output(self,raw_output):
        anchors = self.config['model']['anchors']
        anchor_52 = anchors[0 * 2:3 * 2]
        anchor_26 = anchors[3 * 2:6 * 2]
        anchor_13 = anchors[6 * 2:9 * 2]

        grid = raw_output.shape[1]
        grid_coord = int(416 / grid)
        bboxes_xy = raw_output[..., 0:2]
        bboxes_wh = raw_output[..., 2:4]
        bboxes_xy = sigmoid(bboxes_xy)

        c_mask = np.zeros(shape=bboxes_xy.shape)
        for batch_i in range(c_mask.shape[0]):
            for cy in range(c_mask.shape[1]):
                for cx in range(c_mask.shape[2]):
                    for j in range(c_mask.shape[3]):
                        c_mask[batch_i][cy][cx][j] = np.array([cx * grid_coord, cy * grid_coord])

        bboxes_xy = bboxes_xy + c_mask
        bboxes_wh = np.exp(bboxes_wh)
        # print('bboxes_wh:',bboxes_wh[1,3,7,0])
        anchors_big = np.zeros(shape=bboxes_wh.shape, dtype=np.float32)
        for i in range(3):
            if grid == 13:
                anchors_big[:, :, :, i, :] = np.array(anchor_13[i * 2:(i + 1) * 2])
            elif grid == 26:
                anchors_big[:, :, :, i, :] = np.array(anchor_26[i * 2:(i + 1) * 2])
            elif grid == 52:
                anchors_big[:, :, :, i, :] = np.array(anchor_52[i * 2:(i + 1) * 2])
            else:
                raise Exception('wrong grid size!', grid)

        bboxes_wh = bboxes_wh * anchors_big
        # print('bboxes_wh:',bboxes_wh[1,3,7,1])

        classes_score = self.calc_classes_score(raw_output)
        bboxes = np.concatenate((bboxes_xy, bboxes_wh, classes_score), axis=-1)

        return bboxes
        # sort
    def inference(self,output):
        thresh = self.config['model']['threshold']
        r1 = self.regulize_single_raw_output(output[0])
        r2 = self.regulize_single_raw_output(output[1])
        r3 = self.regulize_single_raw_output(output[2])
        r  = [r1,r2,r3]
        batch_winners = []
        for batch_i in range(r[0].shape[0]):
            # 1.放一起
            all_bboxes = []
            for i in range(3):
                ro = r[i][batch_i]
                for cy in range(ro.shape[0]):
                    for cx in range(ro.shape[1]):
                        for j in range(ro.shape[2]):
                            bbox = ro[cy, cx, j]
                            all_bboxes.append(bbox)

            # 2.按score排序。（以每个bbox里面最大的score记）
            sorted_bboxes = sorted(all_bboxes, key=lambda item: np.max(item[4:]), reverse=True)
            # 2.5 compress 移除class_score小于thresh的bbox
            compressed_sorted_bboxes = []
            for i, box in enumerate(sorted_bboxes):
                if np.max(box[4:]) <= thresh:
                    compressed_sorted_bboxes = sorted_bboxes[0:i]
                    break
            sorted_bboxes = compressed_sorted_bboxes
            print('压缩后的bboxes:', len(compressed_sorted_bboxes))
            # 3. 开始nms
            # final_bboxes = self.do_nms(sorted_bboxes,self.config['model']['nms_iou_threshold'])
            final_bboxes = self.do_tf_nms(sorted_bboxes,self.config['model']['nms_iou_threshold'])
            print('final bboxes count:', len(final_bboxes))
            if self.debug:
                img = np.zeros(shape=[416, 416, 3], dtype=np.uint8)
                after_img = draw_bboxes2(img, final_bboxes)
                cv2.imshow('after img', after_img)
                cv2.waitKey(0)
            batch_winners.append(final_bboxes)
        return batch_winners

    def do_nms(self,sorted_bboxes,thresh):
        nms_thresh = 0.5

        t0 = time.time()
        for i in range(len(sorted_bboxes)):
            rect = convert_rect_from_center_to_four_coord(sorted_bboxes[i][0:4])
            sorted_bboxes[i][0:4] = rect
        t1 = time.time()
        elapsed = t1 - t0

        print('预备耗时:%.3f毫秒' % (elapsed * 1000))

        for i in range(len(sorted_bboxes)):
            t0 = time.time()
            current_king_box = sorted_bboxes[i]
            rect1 = current_king_box[0:4]
            for j in range(i + 1, len(sorted_bboxes)):
                # we have to kill all the challengers that is too near to me, all of them!!!
                challenge_box = sorted_bboxes[j]
                if np.max(challenge_box) > 0:
                    rect2 = challenge_box[0:4]
                    iou_result = iou2(rect1, rect2)
                    if iou_result > nms_thresh:
                        # too near to me, kill on sight!!!
                        sorted_bboxes[j] = np.zeros(shape=sorted_bboxes[0].shape, dtype=np.float32)
                    # else:
                    # too far away from me ,I don't care if it lives.
                    # pass
            t1 = time.time()
            elapsed = t1 - t0
            if i % 10 == 0:
                print('i:%d 耗时:%.3f毫秒' % (i, elapsed * 1000))

        tmp = sorted(sorted_bboxes, key=lambda item: np.max(item[4:]), reverse=True)
        final_bboxes = tmp
        for i, box in enumerate(tmp):
            if np.max(box[4:]) <= thresh:
                final_bboxes = tmp[0:i]
                break
        return final_bboxes
    def do_tf_nms(self,sorted_bboxes,thresh):
        b = np.array(sorted_bboxes)
        coord = b[...,0:4]
        x1 = coord[...,0:1]
        y1 = coord[...,1:2]
        x2 = coord[...,2:3]
        y2 = coord[...,3:4]
        new_coord = np.concatenate([y1,x1,y2,x2],axis=-1)
        scores = np.max(b[...,4:],axis=-1)
        selected_indices = tf.image.non_max_suppression(new_coord,scores,max_output_size=self.config['model']['max_objects_per_image'],
                                     iou_threshold=thresh)
        bboxes_on_image = []
        # 初始化所有variables 的op
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            indices = sess.run(selected_indices)
            for i in range(indices.shape[0]):
                bbox = Bbox()
                bbox.x1 = int(b[indices[i]][0])
                bbox.y1 = int(b[indices[i]][1])
                bbox.x2 = int(b[indices[i]][2])
                bbox.y2 = int(b[indices[i]][3])
                bbox.label = self.config['model']['classes'][np.argmax(b[indices[i]][4:])]
                bbox.confidence = float(np.max(b[indices[i]][4:]))
                bboxes_on_image.append(bbox)
        return bboxes_on_image

    def do_mAP(self,bboxes):
        #计算mAP，输入的bboxes应为经过nms抑制过的最终bboxes输出，即self.inference的输出
        pass
    def do_AP(self):
        pass
def prepare_data(train_folder,val_folder,batch_size):
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       preprocessing_function=lambda x:((x/255)-0.5)*2)
    val_datagen = ImageDataGenerator(rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       preprocessing_function=lambda x:((x/255)-0.5)*2)
    # 默认的color_mode是RGB
    train_generator = train_datagen.flow_from_directory(directory=train_folder,
                                                        target_size=(256,256),batch_size=batch_size)
    val_generator   = val_datagen.flow_from_directory(directory=val_folder,
                                                      target_size=(256,256),batch_size=batch_size)
    return train_generator,val_generator
if __name__ == "__main__":
    pass