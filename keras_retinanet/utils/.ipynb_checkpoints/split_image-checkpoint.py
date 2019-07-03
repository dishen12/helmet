import cv2
import numpy as np
import os

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
            
            split_img=imgs[i*w_num+j]
            shape=(h_end-i*th,w_end-j*tw)
            if split_img.shape[0:2]!=shape:
                new_img=cv2.resize(split_img,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
                image[i*th:h_end,j*tw:w_end]=new_img
            else:
                image[i*th:h_end,j*tw:w_end]=split_img
                
    return image