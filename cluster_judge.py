
import cv2
import os
import numpy as np
import random
from numpy import *


class RegionNode(object):
    def __init__(self, region, idx):
        self.region = region
        self.left = None
        self.right = None
        self.cluster = [idx]

class RegionTree(object):
    def __init__(self):
        self.root = None
        self.count = 0
        self.idx = 0
    
    def insert(self, region):
        self.root = self._insert(self.root, region)
    
    def _insert(self, node, region):
        if node is None:
            node = RegionNode(region, self.idx)
            self.count += 1
            self.idx += 1
            return node
        elif self._intersect(node.region, region):
            node.region[0] = min(node.region[0], region[0])
            node.region[1] = max(node.region[1], region[1])
            node.cluster.append(self.idx)
            self.idx += 1
        elif node.region[0] < region[0]:
            node.left = self._insert(node.left, region)
        else:
            node.right = self._insert(node.right, region)
        return node
    
    @staticmethod
    def _intersect(region1, region2):
        #region1_length, region2_length = region1[1] - region1[0], region2[1] - region2[0]
        #intersection = max(min(region1[1], region2[1]) - max(region1[0], region2[0]), 0)
        #union = max(region1[1], region2[1]) - min(region1[0], region2[0])
        #flag1, flag2 = intersection / union, intersection / min(region1_length, region2_length)
        #return flag1 > 0.6 or flag2 > 0.85
        #return flag1 > 0.7
        return abs((region1[0] + region1[1]) / 2 - (region2[0] + region2[1]) / 2) < 20
    
    def get_nodes(self):
        results = list()
        self._get_nodes(self.root, results)
        return results
    
    @staticmethod
    def _get_nodes(node, nodes):
        if node is None:
            return
        elif node.left is None:
            pass
        else:
            RegionTree._get_nodes(node.left, nodes)
        nodes.append(node)
        RegionTree._get_nodes(node.right, nodes)


class RowRegionTree(RegionTree):
    @staticmethod
    def _intersect(region1, region2):
        return abs((region1[0] + region1[1]) / 2 - (region2[0] + region2[1]) / 2) < 10


class ColRegionTree(RegionTree):
    @staticmethod
    def _intersect(region1, region2):
        return abs(region1[0] - region2[0]) < 50


def cluster(unit_boxes, x_start, y_start):
    unit_boxes.sort(key=lambda e: e[1] + e[3])
    unit_row_boxes = np.array(unit_boxes)
    row_results = list()
    row_regions = [[box[1], box[3]] for box in unit_row_boxes]
    row_tree = RowRegionTree()
    for region in row_regions:
        row_tree.insert(region)
    row_nodes = row_tree.get_nodes()
    for row_node in row_nodes:
        index = row_node.cluster
        row_region = row_node.region
        row_results.append(index)
    unit_boxes.sort(key=lambda e: e[0])
    unit_col_boxes = np.array(unit_boxes)
    col_results = list()
    col_regions = [[box[0], box[2]] for box in unit_col_boxes]
    col_tree = ColRegionTree()
    for region in col_regions:
        col_tree.insert(region)
    col_nodes = col_tree.get_nodes()
    for col_node in col_nodes:
        index = col_node.cluster
        col_region = col_node.region
        col_results.append(index)
    return row_results, col_results

def judge(boxes):
    row_index = cluster(boxes, 0, 0)
    return row_index

# if __name__ == '__main__':
#     image_root = 'D:\\table\\test_Nline_table'
#     image_path = os.path.join(image_root, image_name)
#     img = cv2.imread(image_path)
#     generate_line(boxes, img)
