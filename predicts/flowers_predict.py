from yolo_model import YOLO_V3
import utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
model = load_model("tmp/classification_flowers_ckpt_21_0.98.h5")

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:((x/255)-0.5)*2)

test_generator = test_datagen.flow_from_directory("C:\\All\\Data\\flowers17_tsycnh\\test",
                                                    target_size=(256,256),
                                                    batch_size=1,
                                                    class_mode='categorical')
print("pick 10 random pics")
i=0
for d,l in test_generator:
    r = model.predict(d)
    pred_index = np.argmax(r)
    truth_index = np.argmax(l)
    f =r[0][pred_index]
    if r[0][pred_index]>0.5 and pred_index == truth_index:
        print("gt:%d pred:%d 🍺 acc：%.2f"%(truth_index,pred_index,r[0][pred_index]))
    else:
        print("gt:%d pred:%d 💀"%(truth_index,pred_index))
    i=i+1
    if i ==10:
        break
l,a=model.evaluate_generator(test_generator ,steps=len(test_generator))
print('test acc: ',a)
'''
acc: 96.93%  transfer learning from YOLOv3 with aug (without fine tune)
'''