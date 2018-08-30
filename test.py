import numpy as np
from utils import *
import datetime
from datetime import datetime
import time
np.random.seed(0)
raw_output1 = np.random.rand(4,13,13,3,12)
raw_output2 = np.random.rand(4,26,26,3,12)
raw_output3 = np.random.rand(4,52,52,3,12)
raw_output = [raw_output1,raw_output2,raw_output3]
anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
config = load_json("config_detection.json")
thresh = 0.5# threshold for classes score.


def calc_classes_score(raw_output):
    confidence = raw_output[...,4:5]
    classes = raw_output[...,5:]
    classes_scores = classes*confidence
    # print(raw_output.shape,confidence.shape,classes.shape,classes_scores.shape)
    # print('classes:',classes[0,0,0,0])
    # print('confidence:',confidence[0,0,0,0])
    # print('cls_scores:',classes_scores[0,0,0,0])
    r = np.greater(classes_scores,thresh)
    # print('r:',r[0,0,0,0])
    r2 = np.where(r,classes_scores,0)
    # print('r2:',r2[0,0,0,0])
    return r2

def regulize_single_raw_output(raw_output):
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
            raise Exception('wrong grid size!',grid)


    bboxes_wh = bboxes_wh * anchors_big
    # print('bboxes_wh:',bboxes_wh[1,3,7,1])

    classes_score = calc_classes_score(raw_output)
    bboxes = np.concatenate((bboxes_xy, bboxes_wh, classes_score), axis=-1)

    return bboxes
    # sort

# def compress_r(r):

if __name__ == "__main__":
    # 单output tensor处理
    # NMS应为所有output tensor一起处理
    # fullsizing bbox
    r1 = regulize_single_raw_output(raw_output1)
    r2 = regulize_single_raw_output(raw_output2)
    r3 = regulize_single_raw_output(raw_output3)
    print(r1[1,3,7,1])
    print(r2[1,3,7,1])
    print(r3[1,3,7,1])
    r = [r1,r2,r3]
    # 这里已经将scores处理完毕，所有小于thresh的score都已经搞成0了
    # nms by batch
    def inference():
        batch_winners = []
        for batch_i in range(r[0].shape[0]):
            #1.放一起
            all_bboxes = []
            for i in range(3):
                ro = r[i][batch_i]
                for cy in range(ro.shape[0]):
                    for cx in range(ro.shape[1]):
                        for j in range(ro.shape[2]):
                            bbox = ro[cy,cx,j]
                            all_bboxes.append(bbox)

            #2.按score排序。（以每个bbox里面最大的score记）
            sorted_bboxes = sorted(all_bboxes,key=lambda item:np.max(item[4:]),reverse=True)
            #2.5 compress 移除class_score小于thresh的bbox
            compressed_sorted_bboxes = []
            for i,box in enumerate(sorted_bboxes):
                if np.max(box[4:]) <= thresh:
                    compressed_sorted_bboxes = sorted_bboxes[0:i]
                    break
            sorted_bboxes = compressed_sorted_bboxes
            print('压缩后的bboxes:',len(compressed_sorted_bboxes))
            #3. 开始nms
            nms_thresh = 0.5


            t0 = time.time()
            for i in range(len(sorted_bboxes)):
                rect = convert_rect_from_center_to_four_coord(sorted_bboxes[i][0:4])
                sorted_bboxes[i][0:4]=rect
            t1 = time.time()
            elapsed = t1 - t0

            print('预备耗时:%.3f毫秒' % (elapsed * 1000))

            for i in range(len(sorted_bboxes)):
                # if newest_winner_index >= len(sorted_bboxes)-1:
                #     break

                t0 = time.time()

                current_king_box = sorted_bboxes[i]
                rect1 = current_king_box[0:4]
                for j in range(i+1,len(sorted_bboxes)):
                    # we have to kill all the challengers that is too near to me, all of them!!!
                    # print('i:%d j:%d'%(i,j))
                    # if i==1 and j == 345:
                    #     a=0
                    challenge_box = sorted_bboxes[j]
                    if np.max(challenge_box) > 0:
                        rect2 = challenge_box[0:4]
                        iou_result = iou2(rect1,rect2)
                        if iou_result > nms_thresh:
                            # too near to me, kill on sight!!!
                            sorted_bboxes[j] = np.zeros(shape=sorted_bboxes[0].shape,dtype=np.float32)
                        # else:
                            # too far away from me ,I don't care if it lives.
                            # pass
                t1 = time.time()
                elapsed = t1-t0
                if i%10 == 0:
                    print('i:%d 耗时:%.3f毫秒'%(i,elapsed*1000))

            tmp = sorted(sorted_bboxes,key=lambda item:np.max(item[4:]),reverse=True)
            final_bboxes = tmp
            for i,box in enumerate(tmp):
                if np.max(box[4:]) <= thresh:
                    final_bboxes = tmp[0:i]
                    break
            print('final bboxes count:',len(final_bboxes))
            img = np.zeros(shape=[416,416,3],dtype=np.uint8)
            after_img = draw_bboxes(img,final_bboxes)
            cv2.imshow('after img', after_img)
            cv2.waitKey(0)
            # batch_winners.append(winners)