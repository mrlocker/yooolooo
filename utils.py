import numpy as np
import os
import cv2
import imgaug as ia

def preprocess(x):
    return ((x/255)-0.5)*2

def softmax(list):
    e1 = np.exp(list)
    d2 = np.sum(e1)
    result = e1/np.sum(d2)
    return result

def sigmoid(list):
    return 1/(1 +(1/np.exp(list)))

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
# 判断两矩形是否相交，若相交返回相交面积，否则返回-1,gap表示两个矩形之间如果有gap大小的距离也算相交
def rect_interaction(rect1, rect2, gap=0):
    #  rect1 = [x1,y1,x2,y2]  x1,y1 左上角矩形坐标 x2,y2 右下角矩形坐标
    x1, y1, x2, y2 = rect1[0] - gap, rect1[1] - gap, rect1[2] + gap, rect1[3] + gap
    x3, y3, x4, y4 = rect2[0] - gap, rect2[1] - gap, rect2[2] + gap, rect2[3] + gap

    a = max(x1, x3)
    b = min(x2, x4)
    c = max(y1, y3)
    d = min(y2, y4)

    if a - b <= 0 and c - d <= 0:
        return (b - a) * (d - c)
    else:
        return -1

def iou(rect1,rect2):
    bbox1 = ia.BoundingBox(rect1[0],rect1[1],rect1[2],rect1[3])
    bbox2 = ia.BoundingBox(rect2[0],rect2[1],rect2[2],rect2[3])
    area = rect_interaction(rect1,rect2)
    if area == -1:
        return 0
    else:
        return area/(bbox1.area + bbox2.area -area)
if __name__ == "__main__":
    pass