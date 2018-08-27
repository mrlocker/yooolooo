from yolo_model import YOLO_V3
from utils import *
from data_generator import St_Generator
from keras import backend as K
import tensorflow as tf


def yolo_loss(y_true, y_pred):
    # batch_index,cy,cx,sub_anchor_index,inner_index
    # 1. batch中的数据逐个循环
    print("y_true type:%s, y_pred type:%s" % (type(y_true), type(y_pred)))
    lambda_coord = 5
    lambda_noobj = 0.5

    # 1. prepare y_pred
    y_pred_xy = tf.sigmoid(y_pred[..., 0:2])  # scale x to 0~1
    y_pred_wh = tf.sigmoid(y_pred[..., 2:4])  # scale confidence to 0~1 #（4，13，13，3，1）
    y_pred_confidence = tf.sigmoid(y_pred[...,4:5])
    y_pred_classes = y_pred[...,5:]

    y_true_xy = y_true[..., 0:2]
    y_true_wh = y_true[..., 2:4]
    y_true_confidence = y_true[...,4:5]
    y_true_classes = y_true[...,5:]

    obj_mask = y_true[..., 4:5]  # 1 means the box exists object
    no_obj_mask = tf.subtract(tf.constant(1, shape=obj_mask.get_shape(), dtype=tf.float32), obj_mask)

    # 2. calc xy loss
    xy_minus = tf.subtract(y_true_xy, y_pred_xy)
    xy_square = tf.square(xy_minus)
    xy_sum = tf.reduce_sum(xy_square,axis=-1,keep_dims=True)
    xy_loss = tf.reduce_sum(tf.multiply(xy_sum,obj_mask))
    # 3. calc wh loss
    wh_minus = tf.subtract(tf.sqrt(y_pred_wh),tf.sqrt(y_true_wh))
    wh_square = tf.square(wh_minus)
    wh_sum = tf.reduce_sum(wh_square,axis=-1,keep_dims=True)
    wh_loss = tf.reduce_sum(tf.multiply(wh_sum,obj_mask))
    # 4. calc confidence loss
    con_minus = tf.subtract(y_true_confidence,y_pred_confidence)
    con_square = tf.square(con_minus)
    con_loss = tf.reduce_sum(tf.multiply(con_square,obj_mask))
    no_con_loss = tf.reduce_sum(tf.multiply(con_square,no_obj_mask))
    # 5. calc classes loss
    y_pred_classes_soft = tf.nn.softmax(y_pred_classes,axis=-1)
    classes_minus = tf.subtract(y_true_classes,y_pred_classes_soft)
    classes_square = tf.square(classes_minus)
    classes_loss = tf.reduce_sum(tf.multiply(classes_square,obj_mask))

    # 6.total loss
    final_xy_loss = tf.multiply(xy_loss,tf.convert_to_tensor(lambda_coord,dtype=tf.float32))
    final_wh_loss = tf.multiply(wh_loss,tf.convert_to_tensor(lambda_coord,dtype=tf.float32))
    final_con_loss = con_loss
    final_no_con_loss = tf.multiply(no_con_loss,tf.convert_to_tensor(lambda_noobj,dtype=tf.float32))
    final_classes_loss = classes_loss

    total_loss = tf.add_n([final_xy_loss,final_wh_loss,final_con_loss,final_no_con_loss,final_classes_loss])
    batch_size = y_pred.get_shape()[0].value
    total_loss = tf.divide(total_loss,tf.convert_to_tensor(batch_size,dtype=tf.float32))
    total_loss = tf.Print(total_loss, [total_loss], message='total Loss \t')

    #########
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     print(sess.run([final_xy_loss,final_wh_loss,final_con_loss,final_no_con_loss,final_classes_loss]))
    #     print('total_loss :',sess.run(total_loss))
    #     tf.Print()


    return total_loss


if __name__ == "__main__":
    config = load_json('unit_test/config_test_yolo_loss.json')

    gen = St_Generator(config)
    print('len:', len(gen))
    one_batch = gen.__getitem__(0)
    # yolo = YOLO_V3(config=config)
    yt = one_batch[1][0]
    print('max:',np.max(yt[0]))
    yt2 = one_batch[1][1]
    yt3 = one_batch[1][2]
    yp = np.random.rand(yt.shape[0],yt.shape[1],yt.shape[2],yt.shape[3],yt.shape[4])

    yt = tf.convert_to_tensor(yt,dtype=tf.float32)
    yp = tf.convert_to_tensor(yp,dtype=tf.float32)
    a= yolo_loss(y_true=yt,y_pred=yp)
    print(a)