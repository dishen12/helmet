"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import numpy as np
import os

import cv2
import pickle
import numpy as np
import tensorflow as tf
import requests,json


def split_image(image,target_size,draw_split=False):
    """
    target_size=[th,tw]
    """
    if isinstance(target_size,int):
        target_size=(target_size,target_size)
    
    h,w,c=image.shape
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1
    
    imgs=[]
    draw_img=image
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th
            
            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw
            
            
            if draw_split:
                pt1=(j*tw,i*th)
                pt2=(w_end,h_end)
                draw_img=cv2.rectangle(draw_img,pt1,pt2,color=(255,0,0),thickness=5)
            else:
                img=image[i*th:h_end,j*tw:w_end]
                imgs.append(img)
    
    if draw_split:
        return draw_img
    else:
        return imgs
    
def merge_image(imgs,target_size,origin_size):
    if isinstance(target_size,int):
        target_size=(target_size,target_size)
        
    h,w,c=origin_size
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1
    
    image=np.zeros(origin_size,np.uint8)
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th
            
            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw
            #print("i*w_num+j is",i*w_num+j,"len is ",len(imgs),"h_num is",h_num,"w_num is ",w_num,"size is ",origin_size)
            split_img=imgs[i*w_num+j]
            shape=(h_end-i*th,w_end-j*tw)
            #print("the size of image is ",image.shape,"the size of split image is:",split_img.shape)
            if split_img.shape[0:2]!=shape:
                new_img=cv2.resize(split_img,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
                image[i*th:h_end,j*tw:w_end,:]=new_img
            else:
                image[i*th:h_end,j*tw:w_end,:]=split_img
                
    return image

def nms(bounding_boxes, confidences, threshold=0.85):
    """
    Args:
        bounding_boxes: np.array([(x1, y1, x2, y2), ...])
        confidences: np.array(conf1, conf2, ...),数量需要与bounding box一致,并且一一对应
        threshold: IOU阀值,若两个bounding box的交并比大于该值，则置信度较小的box将会被抑制

    Returns:
        index 
    """
    len_bound = bounding_boxes.shape[0]
    len_conf = confidences.shape[0]
    if len_bound != len_conf:
        raise ValueError("Bounding box 与 Confidence 的数量不一致")
    if len_bound == 0:
        return np.array([]), np.array([])
    bounding_boxes, confidences = bounding_boxes.astype(np.float), np.array(confidences)

    x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    pick = []
    while len(idxs) > 0:
        # 因为idxs是从小到大排列的，last_idx相当于idxs最后一个位置的索引
        last_idx = len(idxs) - 1
        # 取出最大值在数组上的索引
        max_value_idx = idxs[last_idx]
        # 将这个添加到相应索引上
        pick.append(max_value_idx)

        xx1 = np.maximum(x1[max_value_idx], x1[idxs[: last_idx]])
        yy1 = np.maximum(y1[max_value_idx], y1[idxs[: last_idx]])
        xx2 = np.minimum(x2[max_value_idx], x2[idxs[: last_idx]])
        yy2 = np.minimum(y2[max_value_idx], y2[idxs[: last_idx]])

        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

        iou = w * h / areas[idxs[: last_idx]]
        # 删除最大的value,并且删除iou > threshold的bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last_idx], np.where(iou > threshold)[0])))

    return pick

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.
 
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        
        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            #draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_detections

def upload(video_path):
        """
        return example: {"fileIp":"10.50.200.107:8888",
        "fileUrl":"group1/M00/00/00/CjLIa10V-92AUONHAAAACV_xtOE8838105",
        "success":true}
        """
        url='http://222.173.73.19:8762/mtrp/file/json/upload.jhtml'
        with open(video_path,'rb') as file:
            files = {'upload': file}
            r = requests.post(url, files=files)
        #print(r)
        result=json.loads(r.content)
        if (result['success']):
            print("upload success!")
            return result['fileUrl']
        else:
            print('upload',video_path,'failed')
            print('file upload server return',r.content)
            return None

class Stack:
 
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size
 
    def is_empty(self):
        return len(self.items) == 0
 
    def pop(self):
        return self.items.pop()
 
    def peek(self):
        if not self.is_empty():
            return self.items[len(self.items) - 1]
 
    def size(self):
        return len(self.items)

    def clear_all(self):
        #self.items = []
        self.items.clear()
 
    def push(self, item):
        self.items.append(item)      
    
def capture_thread(video_path, frame_buffer, lock):
    print("capture_thread start")
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print('video_path: ' + video_path)
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value is False:
            break
        lock.acquire()
        frame_buffer.push(frame)
        #print("push one")
        lock.release()
        cv2.waitKey(1)

def _get_detections_video(generator,videoPath, model, score_threshold=0.05, max_detections=100, save_path=None):
    import cv2 
    import datetime
    import threading

    """
    connect the mysql
    """
    import mysql.connector
    import json
    config = {"host": "222.173.73.19", "user": "root", "passwd": "xfsdbpasswd", "port": 10020, "database": "XFCMSDB"}
    mydb = mysql.connector.connect(**config)
    cursor = mydb.cursor()
    
    cut_video_id = 0
    videoWriter = None
    video_path = None
    alarmTime = None
    pre_alarm = None
    pre_c_f = 0 
    pre_c_p = 0
    
    #cap=cv2.VideoCapture(videoPath)
    frame_buffer = Stack(1)
    lock = threading.RLock()
    t1 = threading.Thread(target=capture_thread, args=(videoPath, frame_buffer, lock))
    t1.start()
    i = 0
    while(True):
        #print("the size of buffer is :",frame_buffer.size())
        if (frame_buffer.size() > 0):
            
            lock.acquire()
            frame = frame_buffer.pop()
            frame_buffer.clear_all()
            lock.release()
            
            #ret,frame = cap.read()
            #if(ret==False):
                #print("video is interrupted!")
                #break
            h,w,_ = frame.shape
            i += 1
            #print("frame id:",i)
            raw_image    = frame.copy()
            image        = generator.preprocess_image(raw_image.copy())
            image, scale = generator.resize_image(image)

            c_p = 0
            c_f = 0

            # run network
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes      = boxes[0, indices[scores_sort], :]
            image_scores     = scores[scores_sort]
            image_labels     = labels[0, indices[scores_sort]]
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            #print(image_boxes)

            selection = np.where(image_scores > score_threshold)[0]
            for s in selection:
                if(image_labels[s] in (0,1,2,3)):
                    c_p += 1
                else:
                    c_f += 1
            if(i%100==0 and c_f>0):
                #截取视频
                if(videoWriter is not None):
                    videoWriter.release()
                alarmTime = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
                if(video_path is not None):
                    # post the mes
                    url_post(image_boxes,image_scores,image_labels,(h,w),label_to_name=generator.label_to_name)
                    
                    fileUrl = upload(video_path)
                    #插入数据库
                    if(pre_alarm is None): 
                        print("errot")
                        return 
                    event_id = str(i)
                    content = "现场监测到{}个工人，有{}个工人未带安全帽".format(pre_c_p+pre_c_f,pre_c_f)
                    device_id = "46"
                    ip = "222.173.73.19:8761"
                    sql_query = "insert into mtrp_alarm_test(event_id,alarmTime,content,device_id,fileUrl,ip) values('{}','{}','{}','{}','{}','{}')".format(event_id,pre_alarm,content,device_id,fileUrl,ip)
                    try:
                        cursor.execute(sql_query)
                        mydb.commit()
                        print("inser success")
                    except Exception as e:
                        print("insert failed! sql is:",sql_query)
                        print(e)

                pre_c_f = c_f
                pre_c_p = c_p
                if(alarmTime is not None):
                    pre_alarm = alarmTime

                cut_video_id = 0
                out_video_path = "/home/sy/keras-retinanet/wurenji/helmet/outVideo/" #之后这里要改
                video_path = os.path.join(out_video_path,'out_{}.mp4'.format(i))
                videoWriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),20,(w,h))  


            if save_path is not None:
                draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)
                
                #url_post(image_boxes,image_scores,image_labels,(h,w),label_to_name=generator.label_to_name)
                
                if(videoWriter is not None and cut_video_id<100):
                    videoWriter.write(raw_image)
                    print("cut_video_id is ",cut_video_id)
                    cut_video_id += 1
                #yield raw_image,image_boxes,image_scores,image_labels,(h,w)
                yield raw_image,c_p,c_f
    mydb.close()
    return None

def url_post(bboxes,scores,labels,size,label_to_name,score_threshold=0.5):
    
    #data = {"bbox":[[x1,x2,y1,y2],[....]],"conf":[....],"class":[.....],"image_size":[(h,w),(...)]}
    selection = np.where(scores > score_threshold)[0]
    labels_t = [label_to_name(labels[i]) for i in selection]
    d = {"image_size":size,"bbox":bboxes[selection,:],"class":labels_t,"conf":scores[selection]}
    print(bboxes[selection,:])
    r=requests.post("https://crg.wiseom.cn/helmet",data=d)
    try:
        if(r.json()["errmsg"]=="OK"):
            print("post success!")
    except Exception as e:
        print("re:",r.json())
        print(e)
    

def _get_detections_split(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        #if save_path is not None:
            #cv2.imwrite(os.path.join(save_path, '{}_raw_image.png'.format(i)), raw_image)
        size = raw_image.shape
        ## split Image
        imgs = split_image(raw_image,350)
        #temp_imgs = merge_image(imgs,350,size)
        #if save_path is not None:
            #cv2.imwrite(os.path.join(save_path, '{}_init_temp.png'.format(i)), temp_imgs)
        det_imgs = []
        for j,img in enumerate(imgs):
            image        = generator.preprocess_image(img.copy())
            #image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            image, scale = generator.resize_image(image.copy())

            # run network
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            
            #print("the shape of boxes,scores,labels is:",boxes.shape,scores.shape,labels.shape)
            
            # correct boxes for image scale
            boxes /= scale

            
            """old here------------------------------------------"""
            # select indices which have a score above the threshold
            #indices = np.where(scores[0, :] > score_threshold)[0]
            
            #nms for the bbox
            pick = np.array(nms(boxes[0],scores[0],0.95))
            print("the shape of index which has been nmsed ",pick.shape)
            max_detections = max_detections if max_detections < (pick.shape)[0] else (pick.shape)[0]
            indices = pick
            
            # select those scores
            scores = scores[0][indices]
            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]
            # select detections
            image_boxes      = boxes[0, indices[scores_sort], :]
            image_scores     = scores[scores_sort]
            image_labels     = labels[0, indices[scores_sort]]
            draw_detections(img, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)
            #print("box is ",image_boxes)
            #cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(i,j)), img)
            det_imgs.append(img)
            
        out_image = merge_image(det_imgs,350,size)
        if save_path is not None:
            #draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), out_image)

    return None

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    video_path=None,
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    if(video_path is None):
        all_detections     = _get_detections(generator,VideoPath, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    else:
        print("video Path is not None")
        print(video_path)
        _get_detections_video(generator,video_path,model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_detections = None
    
    
    if(all_detections is None): return 0
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions
