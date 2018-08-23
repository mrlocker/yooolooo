import numpy as np
import os
import cv2
def preprocess(x):
    return ((x/255)-0.5)*2

def softmax(list):
    e1 = np.exp(list)
    d2 = np.sum(e1)
    result = e1/np.sum(d2)
    return result

def load_json(path):
    import json
    with open(path,encoding='UTF-8') as config_buffer:
        config = json.loads(config_buffer.read())
        return config
def get_dir_filelist_by_extension(dir, ext):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.split('.')[-1] == ext:
            file_list.append(dir + '/' + item)
    return file_list

def load_anno_xml(xml_path):
    with open(xml_path) as xmlbuffer:
        xml_content = xmlbuffer.read()
def draw_detections(bg, detections, gt,hide_gt=False,hide_confidence=False):
    # detections: x1,y1,x2,y2,c,label
    #            c: 0~1的置信度
    #            label:类别
    # gt: x1,y1,x2,y2,label真值
    pred_color = (255,255,255)
    for rect in detections:
        cv2.rectangle(bg,pt1=(rect[0],rect[1]),pt2=(rect[2],rect[3]),color=pred_color,thickness=2)
        if hide_confidence:
            confidence = ''
        else:
            confidence = str(rect[4])
        cv2.putText(bg,rect[5]+confidence+' ',(rect[0]+2,rect[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,pred_color)
    if not hide_gt:
        for rect in gt:
            cv2.rectangle(bg,pt1=(rect[0],rect[1]),pt2=(rect[2],rect[3]),color=(0,255,0))
            cv2.putText(bg,''+rect[4],(rect[0]+1,rect[3]-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))

    return bg
if __name__ == "__main__":
    a = np.array([2,4,5,7,9])
    b = softmax(a)
    if np.max(b)>0.5:
        index = np.argmax(b)
    pass