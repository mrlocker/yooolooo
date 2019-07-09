##yolov3

### Step1：convert official weights to keras compatible type(.h5)  
download yolov3.weights from https://pjreddie.com/media/files/yolov3.weights   
download yolov3 config file from  https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg  

use convert.py to convert weights format

`tools/convert.py path/to/yolov3.cfg path/to/yolov3.weights outpath/to/yolov3.h5`

### Step2:
Try some classifiction task.
run `trains/train_classification.py` with classification config.
In this experiment, Oxford flowers dataset is used. data can be downloaded here https://download.csdn.net/download/tsyccnh/10641502

run `predicts/flowers_predict.py` to evluate on test set.

The test accuracy should be around 97%
### Step3:
...to be added