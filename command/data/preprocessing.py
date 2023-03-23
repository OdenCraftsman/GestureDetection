from __future__ import annotations
import os
import copy
import numpy as np

def normalization(data:np.array, max_x:int=160, max_y:int=90, min:int=-1) -> np.array:
    list = copy.deepcopy(data)
    for i in range(len(list)):
        for j in range(len(list[i])):
            for k in range(len(list[i][j])):
                list[i][j][k][0] = (list[i][j][k][0]-min)/(max_x - min)
                list[i][j][k][1] = (list[i][j][k][1]-min)/(max_y - min)
    return np.array(list, dtype=np.float64)

def del_legs(data:np.array) -> np.array:
    list = copy.deepcopy(data)
    for mov_num, mov_data in enumerate(list):
        for f, frame_data in enumerate(mov_data):
            list[mov_num][f][8:14] = [-1, -1]
    return list

def coordinate_transformation(data:np.array, width:int=160, height:int=90) -> np.array:
    list = copy.deepcopy(data)
    for mov_num, mov_data in enumerate(data):
        for f, frame_data in enumerate(mov_data):
            for i, pos in enumerate(frame_data):
                if pos[0]==-1 and pos[1]==-1:
                    continue
                if i == 0:
                    base_pos_x = pos[0]
                    base_pos_y = pos[1]
                list[mov_num][f][i][0] = (pos[0]-base_pos_x+width/2)
                list[mov_num][f][i][1] = (pos[1]-base_pos_y+height/2)
    return list

def vector_conversion(data:np.array) -> np.array:
    pass