from yolo_model import YOLO_V3,prepare_data
from keras.optimizers import Adam
import tensorflow as tf

if __name__ == "__main__":
    yolo =YOLO_V3()
    yolo.construct_model(output_units=17)
    yolo.model.compile(optimizer=Adam(), loss=tf.losses.softmax_cross_entropy)

    yolo.load_weights('fl_model.h5')

    _,val_gen =prepare_data()
    result = yolo.model.predict_generator(val_gen)
    # result = yolo.evaluate(val_gen)
    print(result)