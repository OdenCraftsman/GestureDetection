from __future__ import annotations
import os
import copy
import numpy as np

def make_parts_point_list(candidate_list:list, subset_list:list) -> np.array:
    parts_point_list = [ [-1, -1] for _ in range(18) ]
    subset = subset_list[0]
    for j, parts_num in enumerate(subset[:-2]):
        point_list = [-1,-1]
        if parts_num != -1:
            for candidate in candidate_list:
                if parts_num == candidate[3]:
                    point_list = [candidate[0], candidate[1]]
                    break
        parts_point_list[j] = point_list
    return np.array(parts_point_list)

def normalization(data:np.array, max_x:int=160, max_y:int=90, min:int=-1) -> np.array:
    list = copy.deepcopy(data)
    for i in range(len(list)):
        list[i][0] = (data[i][0]-min)/(max_x - min)
        list[i][1] = (data[i][1]-min)/(max_y - min)
    return list

def del_legs(data:np.array) -> np.array:
    list = copy.deepcopy(data)
    list[8:14] = [-1,-1]
    return list

def coordinate_transformation(data:np.array, width:int=160, height:int=90) -> np.array:
    list = copy.deepcopy(data)
    for i, pos in enumerate(data):
        if pos[0]==-1 and pos[1]==-1:
            continue
        if i == 0:
            base_pos_x = pos[0]
            base_pos_y = pos[1]
        list[i][0] = (pos[0]-base_pos_x+width/2)
        list[i][1] = (pos[1]-base_pos_y+height/2)
    return list

def vector_conversion(data:np.array) -> np.array:
    pass