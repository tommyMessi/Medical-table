
import cv2
import json
import numpy as np
import logging
log = logging.getLogger(__name__)
from cluster_judge import judge
import os
from utils import box_get_text, judge_table_type, iou_count, read_json,table_body, k_mean



def post_go(image, text_boxes):
    
    h,w,c = image.shape

    np_tbb = np.array(text_boxes)
    ab = np_tbb[:,:-1].astype(np.float).tolist()
    # cluster without header
    table_body_row_index,table_body_col_index = judge(ab)
    K_body = len(table_body_col_index)

    # select final K
    K = K_body

    col_index_kmean = k_mean(K, text_boxes, False)
    # table_body_header_boxes = sorted(table_body_header_boxes, key=(lambda x:x[0]))
    table_body_header_boxes = np.array(text_boxes)
    #print('table_body_header_boxes:',table_body_header_boxes)
    #sort kmean index
    x_list = []
    for e in col_index_kmean:
        col_box_kmean = table_body_header_boxes[e[0]]
        x_sort = float(col_box_kmean[-1][0])
        x_list.append(x_sort)
    x_list = np.array(x_list)
    x_list_idx = np.argsort(x_list)
    new_col_index_kmean = []
    t = 0
    for x in x_list_idx:
        new_col_index_kmean.append(col_index_kmean[x])
        t = t + 1

    # form result
    result_list = []
    table_body_header_boxes = sorted(table_body_header_boxes, key=(lambda x:float(x[0])))

    return result_list

