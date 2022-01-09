
import os
import cv2
import time
import json
import numpy as np
from io import BytesIO
import codecs
import math
from ocropy import rotate
from table_oritation import oritation_estimate, get_point, rotate_image_by_opencv
from post import post_go

def deskew_orientation(self, image):
    """
    角度纠正
    :param image:
    :return:
    """
    # start_time = time.time()
    image_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(image_encode)
    image_bytes_data = data_encode.tostring()
    image_stream = BytesIO(image_bytes_data)
    image_stream.seek(0)
    image_bytes = image_stream.read()
    image_stream.close()

    deskew_angle = rotate.rotate(image)
    #deskew_angle = 0
    orientation_label = self.orientater.orientation_image_from_bytes(image_bytes)
    orientation_label.label = '11111'
    orientation_angle = 0
    if orientation_label.label == 'vertical':
        orientation_angle = 90
    angle = deskew_angle - orientation_angle
    # print('orientation cost %f ms' % ((time.time() - start_time) * 1000))
    return angle

def table_pre(image, ocr_boxes, table_boxes):

    """
    表格检测
    :param ocr_image:
    :return:
    """
    # start_time = time.time()
    h,w,c = image.shape
    # 创建一个与原图同等大小的空白图
    img_will_angle = np.ones((h,w,c))*255
    boxes_list_detect = []
    # 在空白图画ocr的文本框结果
    for text_area in ocr_boxes:

        topLeftX = text_area[0]
        topLeftY = text_area[1]
        topRightX = text_area[2]
        topRightY = text_area[3]
        bottomRightX = text_area[4]
        bottomRightY = text_area[5]
        bottomLeftX = text_area[6]
        bottomLeftY = text_area[7]
        text = text_area.text['text']
        #print('test_text_0415',text)
        b = np.array([[[topLeftX, topLeftY],[topRightX,topRightY],[bottomRightX, bottomRightY],[bottomLeftX,bottomLeftY]]])
        cv2.fillPoly(img_will_angle, b, (0,0,0))
        boxes_list_detect.append([topLeftX, topLeftY ,topRightX,topRightY,bottomRightX, bottomRightY,bottomLeftX,bottomLeftY])

    boxes_result_json_detect = json.dumps(boxes_list_detect)
    cv2.imwrite('./1.jpg',img_will_angle)

    Ntable_result = []

    V = 0
    for Ntable in table_boxes:
        will_img = image

        table_x1_org = Ntable[0]
        table_y1_org = Ntable[1]
        table_x2_org = Ntable[4]
        table_y2_org = Ntable[5]
        # 把表格位置切出来
        table_need_image = img_will_angle[max(table_y1_org,0):table_y2_org, max(table_x1_org,0):table_x2_org]
        table_need_image = table_need_image.astype(np.float32)
        #print('table_need_image.shape:',table_need_image.shape)
        h,w,c = table_need_image.shape
        # 如果表格太小去除
        if h < 2:
            continue
        # 切出来的图片进行角度估计
        table_ane = deskew_orientation(table_need_image)
        # 反转图片
        inte_rod_img = rotate_image_by_opencv(will_img, table_ane)
        # 此处调用表格解析
        table_cell_dict = post_go(inte_rod_img, ocr_boxes)

        return table_cell_dict