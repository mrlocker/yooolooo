import numpy as np
import os
import cv2
import imgaug as ia

def preprocess(x):
    return ((x/255)-0.5)*2
def afterprocess(x):
    return (x/2+0.5)*255
def softmax(list):
    e1 = np.exp(list)
    d2 = np.sum(e1,axis=-1,keepdims=True)
    result = e1/(d2+1e-6)
    return result

def sigmoid(list):
    return 1/(1 +(1/(np.exp(list)+1e-6)))

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
def area(rect):
    return (rect[2]-rect[0])*(rect[3]-rect[1])
def iou2(rect1,rect2):
    inter_area = rect_interaction(rect1,rect2)
    if inter_area == -1:
        return 0
    else:
        return inter_area/(area(rect1) + area(rect2) -inter_area)

def convert_rect_from_center_to_four_coord(rect):
    x_center = rect[0]
    y_center = rect[1]
    width    = rect[2]
    height   = rect[3]

    x1 = x_center-width/2
    x2 = x_center+width/2
    y1 = y_center-height/2
    y2 = y_center+height/2

    return [x1,y1,x2,y2]

def draw_aug_bboxes(aug_img, bboxes,labels):
    pred_color = (255,255,255)
    for i,rect in enumerate(bboxes.bounding_boxes):
        cv2.rectangle(aug_img, pt1=(rect.x1, rect.y1), pt2=(rect.x2, rect.y2), color=pred_color, thickness=2)
        cv2.putText(aug_img, labels[i] + ' ', (rect.x1 + 2, rect.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color)
        cv2.circle(aug_img,center=(int(rect.center_x),int(rect.center_y)),radius=1,color=(0,0,255),thickness=-1)

    return aug_img
def draw_bboxes(img, bboxes):
    # bboxes are a np array first 4 elems are x1,y1,x2,y2
    pred_color = (255,255,255)
    for i,rect in enumerate(bboxes):
        cv2.rectangle(img, pt1=(int(rect[0]), int(rect[1])), pt2=(int(rect[2]), int(rect[3])), color=pred_color, thickness=1)
        # cv2.putText(img, labels[i] + ' ', (rect.x1 + 2, rect.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color)
        # cv2.circle(img, center=(int(rect.center_x), int(rect.center_y)), radius=1, color=(0, 0, 255), thickness=-1)
    return img
def draw_bboxes2(img, bboxes,color=(255,255,255)):
    for i,bbox in enumerate(bboxes):
        cv2.rectangle(img, pt1=(int(bbox.x1), int(bbox.y1)), pt2=(int(bbox.x2), int(bbox.y2)), color=color, thickness=1)
        cv2.putText(img, "%s %.2f"%(bbox.label,bbox.confidence), (bbox.x1 + 2, bbox.y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
    return img

def single_class_gt_detection_match(gts,detections):
    for i in range(len(detections)):
        detections[i].match = False
    detections = sorted(detections,key= lambda item:item.confidence,reverse=True)
    for gt in gts:
        rect1 = [gt.x1,gt.y1,gt.x2,gt.y2]
        for i,dt in enumerate(detections):
            rect2 = [dt.x1,dt.y1,dt.x2,dt.y2]
            # print("iou:",iou2(rect1, rect2))
            if iou2(rect1,rect2) > 0.5:
                detections[i].match = True
                break
    rects = [[d.confidence,d.match] for d in detections]
    return rects

def get_single_image_gt_count(gts,all_labels):
    gtc = {}
    for label in all_labels:
        gtc[label] = 0
    for gt in gts:
        gtc[gt.label] += 1
    return gtc
def single_image_detection_evaluate(gts,detections,all_labels):
    r = {}
    for current_label in all_labels:
        gts_current_label = []
        detections_current_label = []
        for gt in gts:
            if gt.label == current_label:
                gts_current_label.append(gt)
        for detection in detections:
            if detection.label == current_label:
                detections_current_label.append(detection)

        # 这两个变量存储这当前类别内所有的bbox
        match_result = single_class_gt_detection_match(gts_current_label,detections_current_label)
        r[current_label] = match_result
    gtc = get_single_image_gt_count(gts,all_labels)
    return r,gtc


def ap(detections, gt_count):
    # calc average precision based on pascal voc 2007 standard
    # this ap is only for one class.
    # detections (list) : a n*2 list. first col is confidence second col is True or False of this detections
    # gt_count (int) : total gound truths of this kind

    # sort by confidence ascending
    d = sorted(detections, key=lambda item: item[0], reverse=True)
    detections = np.array(d)
    precisions_recalls = np.zeros(shape=detections[..., 0:2].shape)
    # calc precisions and recalls by rank.ie: top x detections
    for i in range(detections.shape[0]):
        TP = np.sum(detections[0:i + 1, 1:2])  # calc all Trues above this rank
        recall = TP / gt_count
        precision = TP / (i + 1)
        precisions_recalls[i] = [precision, recall]
    # calc recall-precision curve
    r_recall_max_precisions = np.zeros(shape=(11, 2))
    for r in range(11):
        r_recall = r / 10
        for i in range(len(precisions_recalls)):
            if precisions_recalls[i, 1] > r_recall:
                max_precision = np.max(precisions_recalls[i:, 0])
                r_recall_max_precisions[r] = [r_recall, max_precision]
                break
            r_recall_max_precisions[r] = [r_recall, 0]
    ap = np.sum(r_recall_max_precisions[..., 1]) / 11
    return ap


if __name__ == "__main__":
    from yolo_model import *


    def test_single_class_gt_detection_match():
        gts = [ Bbox([10,20,30,40],"WaterDrop",confidence=-1),
        Bbox([100,200,150,275],"WaterDrop",confidence=-1),
        Bbox([50,50,300,90],"WaterDrop",confidence=-1),
        Bbox([300,240,310,340],"WaterDrop",confidence=-1)]

        dts = [ Bbox([15,10,28,35],"WaterDrop",confidence=0.93),
        Bbox([108,208,150,275],"WaterDrop",confidence=0.97),
        Bbox([102,202,151,276],"WaterDrop",confidence=0.96),
        Bbox([110,110,146,140],"WaterDrop",confidence=0.8),
        Bbox([301,241,311,341],"WaterDrop",confidence=0.86),
                Bbox([10, 30, 68, 65], "WaterDrop", confidence=0.93)]

        bg = np.zeros(shape=(416,416,3),dtype=np.uint8)
        img_gt = draw_bboxes2(bg,gts,(0,255,0))
        img_gt_dt = draw_bboxes2(img_gt,dts,(0,0,255))


        r = single_class_gt_detection_match(gts,dts)
        cv2.imshow('gtdt',img_gt_dt)
        cv2.waitKey(0)
    def test_get_single_image_gt_count():
        gts = [ Bbox([10,20,30,40],"WaterDrop",confidence=-1),
        Bbox([100,200,150,275],"WaterDrop",confidence=-1),
        Bbox([50,50,300,90],"Upwarping",confidence=-1),
        Bbox([300,240,310,340],"WaterDrop",confidence=-1)]
        get_single_image_gt_count(gts,[
            "WaterDrop",
            "Upwarping",
            "HorizontalCrack",
            "Scale",
            "RollMark",
            "WaterStain",
            "VerticalCrack"
        ])
    test_get_single_image_gt_count()