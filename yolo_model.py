import keras
from keras.layers import Conv2D,Input,BatchNormalization,\
    LeakyReLU,Add,GlobalAveragePooling2D,Dense,Activation,\
    UpSampling2D,Concatenate,Reshape,Dropout
from keras.models import Model,load_model
from keras.optimizers import Adam,RMSprop
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
        # 这里继续将输出的tensor进行处理，处理完过后要和data generator 尺寸一模一样
        return shit
class Bbox():
    def __init__(self,coord=(0,0,0,0),label="Undefined",confidence = 0):
        self.x1 = coord[0]
        self.y1 = coord[1]
        self.x2 = coord[2]
        self.y2 = coord[3]
        self.label = label
        self.confidence = confidence   # -1 means ground truth
class YOLO_V3():
    def __init__(self,config):
        self.config = config
        self.shits = []
        self.input_size = self.config['model']['image_size']
        self.inputs = Input(shape=(self.config['model']['image_size'][0], self.config['model']['image_size'][1], 3))
        self.num_classes = len(self.config['model']['classes'])
        self.classes = self.config['model']['classes']
        self.batch_size = self.config['model']['batch_size']
        self.debug = self.config['model']['debug']
        # yolo算法采用前后端分离。后端指的是主干网络。主干网络配合不同的前端，可以实现分类或者检测的目的。
        self.construct_backbone(self.inputs)
        # 载入预训练模型参数（仅主干网络）
        if self.config['model']['type'] == "classification":
            self.construct_classification_model()
        elif self.config['model']['type'] == "detection":
            self.construct_detection_model()
            self.anchors = self.config['model']['anchors']
        self.load_pretrain_weights()
        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        self.sess = sess

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
    def load_weights(self,path):
        self.model.load_weights(path)
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
        return None

####-------------------- classification start ---------------####
    def construct_classification_model(self):
        self.shits.append(GlobalAveragePooling2D()(self.shits[-1]))
        output_units = self.config['model']['output_units']
        # self.shits.append(Dropout(0.5)(self.shits[-1]))

        logits = Dense(units=output_units,activation="softmax")(self.shits[-1])
        self.model = Model(inputs=self.inputs,outputs=logits)
        print("分类网络组建完毕")
    def train(self,train_generator,val_generator,log_dir):
        config = self.config
        filepath = "./tmp/classification_flowers_ckpt_{epoch:02d}_{val_acc:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc',
                                     verbose=1, save_best_only=False)
        #
        # def lr_sch(epoch):
        #     # 200 total
        #     if epoch < 50:
        #         return 1e-3
        #     if 50 <= epoch < 100:
        #         return 1e-4
        #     if epoch >= 100:
        #         return 1e-5

        # lr_scheduler = LearningRateScheduler(lr_sch)
        # lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
        #                                mode='max', min_lr=1e-6)
        tb = TensorBoard(log_dir=log_dir,write_graph=False)
        callbacks = [checkpoint,tb]
        self.model.compile(optimizer=RMSprop(1e-4),loss="categorical_crossentropy",metrics=['acc'])
        self.model.summary()

        self.model.fit_generator(generator=train_generator,
                                 epochs=config['train']['epochs'],
                                 steps_per_epoch=len(train_generator),
                                 validation_data=val_generator,
                                 validation_steps=len(val_generator),
                                 class_weight='auto',
                                 callbacks=callbacks)
        self.model.save_weights('fl_model.h5')
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
        r = result
        cv2.imshow('result', image_for_cv2_show)
        pred_index = np.argmax(r)
        pred = self.config['model']['classes'][pred_index]


        if np.max(r) > 0.5:
            if pred == class_name:
                message = " 🍺"
            else:
                message = " 💀"
            print('真值种类名称：%s 预测类别名称：%s %s'%(class_name,pred,message))
        else:
            print('真值：',class_name,'预测值：无'," 💀")
        cv2.waitKey(0)
####-------------------- classification end   ---------------####

####-------------------- detection start      ---------------####
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

    def prepare_cxcy(self,feats):
            feats_shape = tf.shape(feats)[1:3]

            grid_shape_y = feats_shape[0]
            grid_shape_x = feats_shape[1]

            range_y = tf.range(start=0, limit=grid_shape_y)  # [0,1,2,...,11,12]
            range_y = tf.expand_dims(range_y, axis=-1)  # [[0],[1],[2],...,[11],[12]]
            for _ in range(2):  # add to dimension of 4
                range_y = tf.expand_dims(range_y, axis=-1)
            # print(range_y)
            tile_y = tf.tile(range_y, [1, grid_shape_x, 1, 1])
            tile_y = tf.tile(tile_y, [1, 1, 3, 1])
            #
            range_x = tf.range(start=0, limit=grid_shape_x)
            range_x = tf.expand_dims(range_x, axis=0)
            for _ in range(2):  # add to dimension of 4
                range_x = tf.expand_dims(range_x, axis=-1)
            tile_x = tf.tile(range_x, [grid_shape_y, 1, 1, 1])
            tile_x = tf.tile(tile_x, [1, 1, 3, 1])
            #
            anchors_xy = tf.concat([tile_x, tile_y], axis=-1)
            anchors_xy = tf.cast(anchors_xy, dtype=tf.float32)

            # # --------------------
            # sess = tf.Session()
            # sess.run(tf.global_variables_initializer())
            # # ==========
            # data2 = np.ones(shape=(7, 13, 17, 3, 2), dtype=np.float32)
            # print(range_y.eval(session=sess,feed_dict={feats:data2}))
            # print(tile_y.eval(session=sess,feed_dict={feats:data2}))
            # print(tile_x.eval(session=sess,feed_dict={feats:data2}))
            # print(anchors_xy.eval(session=sess,feed_dict={feats:data2}))
            # sess.close()
            return anchors_xy
    def prepare_anchors_wh(self,whfeats):

            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 13*13,26*26,52*52
            grids = whfeats.shape.dims[1].value
            cell_size = int(self.config['model']['image_size'][0] / grids)

            mask_index = int((grids / 13) // 2)

            grid_shape = tf.shape(whfeats)[1:3]
            grid_shape_y = grid_shape[0]
            grid_shape_x = grid_shape[1]

            three_scaled_anchors = []
            for i in range(3):
                anchor_index = anchor_mask[mask_index][i] * 2
                current_anchor = self.anchors[anchor_index:anchor_index + 2]
                current_scaled_anchor = np.array(current_anchor) / cell_size
                three_scaled_anchors.append(current_scaled_anchor)

            three_scaled_anchors = tf.convert_to_tensor(np.array(three_scaled_anchors))  # 3x2
            three_scaled_anchors = tf.expand_dims(three_scaled_anchors, axis=0)  # 1x3x2
            three_scaled_anchors = tf.expand_dims(three_scaled_anchors, axis=0)  # 1x1x3x2

            p_wh = tf.tile(three_scaled_anchors, [grid_shape_y, grid_shape_x, 1, 1])
            p_wh = tf.cast(p_wh,tf.float32)
            return p_wh  # shape:(grdi_shape_y,grid_shape_x,3,2)
            # sess = tf.Session()
            # sess.run(tf.global_variables_initializer())
            #
            # data = np.ones(shape=(7, 13, 17, 3, 2), dtype=np.float32)
            #
            # print(tile_anchors.eval(session=sess,feed_dict={whfeats:data}))
            # print(tile_anchors.eval(session=sess,feed_dict={whfeats:data}).shape)

    def yolo_loss(self,y_true, y_pred):
        # batch_index,cy,cx,sub_anchor_index,inner_index
        # 1. batch中的数据逐个循环
        print("y_true type:%s, y_pred type:%s" % (type(y_true), type(y_pred)))
        lambda_coord = 5
        lambda_noobj = 0.5

        # 1. prepare y_pred
        # 1.1 prepare y_pred_xy
        y_pred_xy = tf.sigmoid(y_pred[..., 0:2])  # scale x to 0~1
        cxcy = self.prepare_cxcy(y_pred) # shape: (13,13,3,2)
        print("y_pred_xy shape:",y_pred_xy.shape)
        print("cxcy shape:",cxcy.shape)
        y_pred_xy += cxcy
        # 1.2 prepare y_pred_wh
        y_pred_wh = tf.exp(tf.sigmoid(y_pred[..., 2:4]))  # scale confidence to 0~1 #（4，13，13，3，1）
        p_wh = self.prepare_anchors_wh(y_pred_wh)
        y_pred_wh = y_pred_wh * p_wh
        # 1.3 prepare y_pred confidence
        y_pred_confidence = tf.sigmoid(y_pred[..., 4:5])
        # 1.4 prepare y_pred classes
        y_pred_classes = y_pred[..., 5:]
        y_pred_classes_soft = tf.nn.softmax(y_pred_classes, axis=-1)

        # prepare y_true
        y_true_xy = y_true[..., 0:2]
        y_true_wh = y_true[..., 2:4]
        y_true_confidence = y_true[..., 4:5]
        y_true_classes = y_true[..., 5:]

        obj_mask = y_true[..., 4:5]  # 1 means the box exists object
        no_obj_mask = tf.subtract(tf.ones_like(obj_mask,dtype=tf.float32), obj_mask)

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
        # print("total loss shape: ",total_loss.get_shape())
        batch_size = tf.shape(y_pred)[0]
        batch_size = tf.cast(batch_size,dtype=tf.float32)
        total_loss /= batch_size

        # total_loss = tf.Print(total_loss, [total_loss], message='total Loss \t')

        #########
        # init_op = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #
        #     print(sess.run([final_xy_loss,final_wh_loss,final_con_loss,final_no_con_loss,final_classes_loss]))
        #     print('total_loss :',sess.run(total_loss))
        #     # tf.Print()

        return total_loss

    def train_detection(self,train_generator,val_generator):
        filepath = "./tmp/detection_ckpt_{epoch:02d}_{loss:.2f}.h5"

        checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss',
                                     verbose=1, save_best_only=False)
        def lr_sch(epoch):
            # 200 total
            lr = self.config['train']['learning_rate']
            total_epochs = self.config['train']['epochs']
            if epoch < total_epochs/3:
                return lr
            if total_epochs/3 <= epoch < total_epochs/3*2:
                return lr*0.1
            if epoch >= total_epochs/3*2:
                return lr*0.01
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
                                       mode='min', min_lr=1e-6)
        lr_scheduler = LearningRateScheduler(lr_sch)

        tb = TensorBoard(log_dir=self.config['train']['log_dir'],write_graph=False)

        self.model.compile(optimizer=Adam(lr=self.config['train']['learning_rate'],
                                          beta_1=0.9,beta_2=0.999),
                           loss=[self.yolo_loss,self.yolo_loss,self.yolo_loss])
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=1*len(train_generator),
                                 epochs=self.config['train']['epochs'],
                                 callbacks=[checkpoint,lr_scheduler,tb])

    def evaluate(self,generator):

        print('开始评估模型性能')
        self.load_weights(self.config['model']['final_model_weights'])# TODO:临时注释，正式版需取消注释
        print('载入完整已训练模型')
        # pr_result = self.model.predict_generator(generator)# 这个函数暂时不用，因为无法获取ground truth
        total_steps = len(generator)

        cls_result = {}
        gt_counts = {}
        for c in self.classes:
            cls_result[c]=[]
            gt_counts[c] = 0
        for i in range(total_steps):
            # step1. 生成一批图像数据，x为图像，均已归一化至-1~1。y为label。x，y，w，h均为归一化的值。即将原图
            # 转化为13*13后原始坐标按比例缩放后的大小。
            x,y = generator.__getitem__(i)
            bboxes = generator.aug_bbses
            labels = generator.aug_labels
            # step2. 网络预测，给出粗糙输出
            raw_outputs = self.model.predict_on_batch(x)
            # step3. 解码粗糙输出，x,y,w,h均作了计算。这里输出的内容应该和data generator输出的ybatch一样
            decoded_outputs = self.decode_raw_output(raw_outputs)
            # step4. 这里最终输出的应该是ready for draw 的bbox
            winners = self.inference(decoded_outputs)
            batch_size = len(bboxes)

            for j in range(batch_size):# iterate each image
                # detcetion_result = np.zeros(shape=(len(winners[j]),2))
                gts = []
                for k in range(len(labels[j])):  # current image's bbox labels
                    current_bbox = bboxes[j].bounding_boxes[k]
                    current_label = labels[j][k]
                    b = Bbox()
                    b.label = current_label
                    b.x1 = current_bbox.x1
                    b.y1 = current_bbox.y1
                    b.x2 = current_bbox.x2
                    b.y2 = current_bbox.y2
                    b.confidence = -1 # means ground truth
                    gts.append(b)
                detections = winners[j]
                if self.debug:
                    origin_img = afterprocess(x[j])
                    origin_img = origin_img.astype(np.uint8)
                    img = draw_bboxes2(origin_img,gts,color=(0,255,0))#gts
                    img2 = draw_bboxes2(img,detections,color=(255,0,0))#detections
                    cv2.imshow('detection result',img2)
                    cv2.waitKey(0)
                r,gt_count = single_image_detection_evaluate(gts,detections,self.classes)
                for c in self.classes:
                    cls_result[c].extend(r[c])
                    gt_counts[c] += gt_count[c]
        aps = self.do_AP(cls_result,gt_counts)
        self.do_mAP(aps)
        # ap()
                    # cls_result[c].append(bboxes[j][k])
            # cls_result[c].append()
        # list = [[ndarray][ndarray][ndarray]] 464*13*13*3*(5+7) 464*26*26*3*(5+7) 464*52*52*3*(5+7)
        # for i in range(pr_result[0].shape[0]):
        #     out0 = pr_result[0][i]
        #     out1 = pr_result[1][i]
        #     out2 = pr_result[2][i]
        #     out = [out0,out1,out2]
        #     result = self.inference([out])[0]# inference为按照单个batch计算的。
        # pr0 = pr_result[0][0]
        # pr1 = pr_result[1][0]
        # pr2 = pr_result[2][0]
        # winners = self.inference([pr0,pr1,pr2])
        # 我需要GRound Truth！！！！！！
        pass

    def calc_classes_score(self,raw_output):
        confidence = sigmoid(raw_output[..., 4:5])
        classes = softmax(raw_output[..., 5:])
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
        # 规范化输出，输出为 [x,y,w,h,classes....]。x,y,w,h均为真实尺寸
        if len(raw_output.shape)<5:
            raw_output = np.expand_dims(raw_output,axis=0)
        # anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 13*13,26*26,52*52

        anchors = self.config['model']['anchors']
        # anchor_52 = anchors[0 * 2:3 * 2]
        # anchor_26 = anchors[3 * 2:6 * 2]
        # anchor_13 = anchors[6 * 2:9 * 2]

        grid = raw_output.shape[1]
        # grid_coord = int(self.input_size[0] / grid)
        bboxes_xy = raw_output[..., 0:2]
        bboxes_wh = raw_output[..., 2:4]
        cxcy = self.prepare_cxcy(bboxes_xy)
        p_wh = self.prepare_anchors_wh(tf.convert_to_tensor(bboxes_wh))
        #with self.sess as sess:
            # sess.run(tf.global_variables_initializer())
        cxcy = self.sess.run(cxcy)
        p_wh = self.sess.run(p_wh)
        bboxes_xy = sigmoid(bboxes_xy)+cxcy
        # bboxes_wh = np.exp(bboxes_wh)*p_wh # old without sigmoid
        bboxes_wh = np.exp(sigmoid(bboxes_wh))*p_wh

        # c_mask = np.zeros(shape=bboxes_xy.shape)
        # for batch_i in range(c_mask.shape[0]):
        #     for cy in range(c_mask.shape[1]):
        #         for cx in range(c_mask.shape[2]):
        #             for j in range(c_mask.shape[3]):
        #                 c_mask[batch_i][cy][cx][j] = np.array([cx * grid_coord, cy * grid_coord])
        #
        # bboxes_xy = bboxes_xy + c_mask
        # bboxes_wh = np.exp(bboxes_wh)
        # print('bboxes_wh:',bboxes_wh[1,3,7,0])
        # anchors_big = np.zeros(shape=bboxes_wh.shape, dtype=np.float32)
        # for i in range(3):
        #     if grid == 13:
        #         anchors_big[:, :, :, i, :] = np.array(anchor_13[i * 2:(i + 1) * 2])
        #     elif grid == 26:
        #         anchors_big[:, :, :, i, :] = np.array(anchor_26[i * 2:(i + 1) * 2])
        #     elif grid == 52:
        #         anchors_big[:, :, :, i, :] = np.array(anchor_52[i * 2:(i + 1) * 2])
        #     else:
        #         raise Exception('wrong grid size!', grid)
        #
        # bboxes_wh = bboxes_wh * anchors_big
        # print('bboxes_wh:',bboxes_wh[1,3,7,1])
        # scale to real world size
        # cell_size = self.input_size[0]/grid
        # bboxes_xy *= cell_size
        # bboxes_wh *= cell_size
        classes_score = self.calc_classes_score(raw_output)
        bboxes = np.concatenate((bboxes_xy, bboxes_wh, classes_score), axis=-1)

        return bboxes
        # sort

    def decode_raw_output(self,raw_output):
        # 输入output为一个batch的数据
        r1 = self.regulize_single_raw_output(raw_output[0])
        r2 = self.regulize_single_raw_output(raw_output[1])
        r3 = self.regulize_single_raw_output(raw_output[2])
        r  = [r1,r2,r3]
        return r
    def inference(self, decoded_output):
        # 输入output为一个batch的数据
        thresh = self.config['model']['threshold']
        r = decoded_output
        batch_winners = []
        grid0 = decoded_output[0].shape[1]
        grid1 = decoded_output[1].shape[1]
        grid2 = decoded_output[2].shape[1]

        cell_size0 = self.input_size[0] / grid0
        cell_size1 = self.input_size[0] / grid1
        cell_size2 = self.input_size[0] / grid2
        # bboxes_xy *= cell_size
        # bboxes_wh *= cell_size

        r[0][...,0:4] *= cell_size0
        r[1][...,0:4] *= cell_size1
        r[2][...,0:4] *= cell_size2

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
            print('原始输出的bboxes个数:', len(sorted_bboxes))

            # 2.5 compress 移除class_score小于thresh的bbox
            compressed_sorted_bboxes = []
            for i, box in enumerate(sorted_bboxes):
                if np.max(box[4:]) <= thresh:
                    compressed_sorted_bboxes = sorted_bboxes[0:i]
                    break
            sorted_bboxes = compressed_sorted_bboxes
            print('压缩后的bboxes个数:', len(compressed_sorted_bboxes))
            # 3. 开始nms
            # final_bboxes = self.do_nms(sorted_bboxes,self.config['model']['nms_iou_threshold'])
            final_bboxes = self.do_tf_nms(sorted_bboxes,self.config['model']['nms_iou_threshold'])
            print('NMS过后的bboxes个数:', len(final_bboxes))
            # if self.debug:
            #     img = np.zeros(shape=[self.input_size[0], self.input_size[1], 3], dtype=np.uint8)
            #     after_img = draw_bboxes2(img, final_bboxes)
            #     cv2.imshow('after img', after_img)
            #     cv2.waitKey(0)
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
        # sorted bboxes 格式是center_x,center_y,width,height
        if len(sorted_bboxes)==0:
            return []
        b = np.array(sorted_bboxes)
        coord = b[...,0:4]
        center_x = coord[...,0:1]
        center_y = coord[...,1:2]
        width = coord[...,2:3]
        height = coord[...,3:4]
        x1 = center_x - width/2
        x2 = center_x + width/2
        y1 = center_y - height/2
        y2 = center_y + height/2
        new_coord = np.concatenate([y1,x1,y2,x2],axis=-1)
        scores = np.max(b[...,4:],axis=-1)
        selected_indices = tf.image.non_max_suppression(new_coord,scores,max_output_size=self.config['model']['max_objects_per_image'],
                                     iou_threshold=thresh)
        bboxes_on_image = []

        indices = self.sess.run(selected_indices)
        for i in range(indices.shape[0]):
            bbox = Bbox()
            center_x = b[indices[i]][0]
            center_y = b[indices[i]][1]
            width = b[indices[i]][2]
            height = b[indices[i]][3]
            bbox.x1 = int(center_x-width/2)
            bbox.y1 = int(center_y-height/2)
            bbox.x2 = int(center_x+width/2)
            bbox.y2 = int(center_y+height/2)
            bbox.label = self.config['model']['classes'][np.argmax(b[indices[i]][4:])]
            bbox.confidence = float(np.max(b[indices[i]][4:]))
            bboxes_on_image.append(bbox)
        return bboxes_on_image
    def do_mAP(self,aps):
        #计算mAP，输入的bboxes应为经过nms抑制过的最终bboxes输出，即self.inference的输出
        count = 0
        sum = 0
        for key in aps:
            sum+=aps[key]
            count+=1
        mAP = sum/count
        print("mAP: ",mAP)
        return mAP
    def do_AP(self,bboxes,gt_counts):
        all_aps = {}
        for c in self.classes:
            single_class_match = bboxes[c]
            single_class_gt_count = gt_counts[c]
            average_precision = ap(single_class_match,single_class_gt_count)
            print("%s AP: %f"%(c,average_precision))
            all_aps[c] = average_precision
        return all_aps
####-------------------- detection end        ---------------####

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
    val_datagen = ImageDataGenerator(preprocessing_function=lambda x:((x/255)-0.5)*2)
    # 默认的color_mode是RGB
    train_generator = train_datagen.flow_from_directory(directory=train_folder,
                                                        target_size=(256,256),batch_size=batch_size)
    val_generator   = val_datagen.flow_from_directory(directory=val_folder,
                                                      target_size=(256,256),batch_size=batch_size)
    return train_generator,val_generator
if __name__ == "__main__":
    pass