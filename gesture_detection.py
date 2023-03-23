from __future__ import annotations
from pathlib import Path

from typing import Optional
import logging

import os
import json
import time
import datetime
import numpy as np
import cv2

from command.data.split_mov import split_all_mov
from command.data.make_dataset import Dataset
from command.preprocessing import *
from command.model.basic_model import TransformerModel

from command.data.src import util
from command.data.src.body import Body

class GestureDetection:
    def __init__(
            self,
            resize_shape:tuple,
            model:TransformerModel,
            base_path: Path,
            openpose_model:Optional[str] = None,
            detection_time:int=3,
            basic_fps:int=30,
            ajust_fps:int=10,
            **setting
        ) -> None:
        self.setting = setting
        self.setting['resize_shape'] = resize_shape
        if openpose_model is None:
            self.setting['openpose_model'] = str(base_path / Path("model/openpose/body_pose_model.pth"))
        else:
            self.setting['openpose_model'] = openpose_model
        self.setting['detection_time'] = detection_time
        self.setting['basic_fps'] = basic_fps
        self.setting['ajust_fps'] = ajust_fps
        self.setting['detection_frame'] = ajust_fps*detection_time
        self.detection_frame_counter = int(basic_fps/ajust_fps)
        self.trans_model = model
        self.body_estimation = Body(self.setting['openpose_model'])
        self.predict_list = []
        self.counter = 1
    def judgement_gesture(self, frame:np.array, preview:bool=False) -> tuple:
        predict = ''
        preview_images = None
        predict_frame = cv2.resize(frame, dsize=self.setting['resize_shape'])
        candidate, subset = self.body_estimation(copy.deepcopy(predict_frame))
        if len(subset) == 0:
            return predict
        parts_point_list = make_parts_point_list(candidate_list=candidate, subset_list=subset)
        parts_point_list = del_legs(parts_point_list)
        parts_point_list = coordinate_transformation(
            data=parts_point_list,
            width=self.setting['resize_shape'][0],
            height=self.setting['resize_shape'][1]
        )
        # --------------- preview ---------------
        if preview:
            preview_images = []
            preview_images.append(copy.deepcopy(predict_frame))
            skeleton_image = util.draw_bodypose(copy.deepcopy(predict_frame), candidate, subset)
            preview_images.append(skeleton_image)
            visualize_data = self._review_dataset(
                data=parts_point_list,
                width=self.setting['resize_shape'][0],
                height=self.setting['resize_shape'][1]
            )
            preview_images.append(visualize_data)
        # --------------- preview ---------------
        parts_point_list = normalization(
            data=parts_point_list,
            max_x=self.setting['resize_shape'][0],
            max_y=self.setting['resize_shape'][1]
        )
        if self.counter == self.detection_frame_counter:
            self.predict_list.append(parts_point_list)
            if np.array(self.predict_list).shape[0] >= self.setting['detection_frame']:
                predict, result_list = self.trans_model.detection(np.array(self.predict_list)[np.newaxis, int(-self.setting['detection_frame']):])
            self.counter = 0
        self.counter += 1
        return predict, preview_images
    def _review_dataset(self, data:np.array, width:int=160, height:int=90) -> np.array:
        base = np.zeros((height, width, 3))
        base += 255
        for pos in data:
            x, y = pos[0], pos[1]
            if x!=-1 and y!=-1:
                cv2.circle(base, (int(x),int(y)), 1, color=(255,0,0), thickness=-1)
        return base


if __name__ == '__main__':
    # 固定変数指定
    SPLIT_TIME = 3
    BASIC_SIZE = 10
    WIDTH_RATE = 16
    HEIGHT_RATE = 16
    WIDTH = BASIC_SIZE*WIDTH_RATE
    HEIGHT = BASIC_SIZE*HEIGHT_RATE
    AJUST_FPS = 10
    MODEL_NAME = 'OURAI'
    EPOCHS = 200
    BATCH_SIZE = 64
    DATASET_SPLIT_RATE = 0.2

    base_path = Path(__file__).parent

    # モデルロード
    print(f"[{datetime.datetime.now()}] start: load model")
    model = TransformerModel(base_path) # must
    model.load_model()
    model.summary()
    print(f"[{datetime.datetime.now()}] end: load model")

    # ジェスチャー検出器の生成
    print(f"[{datetime.datetime.now()}] start: make predicter")
    predicter = GestureDetection(
        resize_shape=(WIDTH, HEIGHT),
        model=model,
        base_path=base_path
    ) # must
    print(f"[{datetime.datetime.now()}] end: make predicter")

    print(f"[{datetime.datetime.now()}] start: load cap")
    cap = cv2.VideoCapture(0)
    predict = ""
    print(f"[{datetime.datetime.now()}] end: load cap")
    print(f"[{datetime.datetime.now()}] start: judgement")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predict, preview_images = predicter.judgement_gesture(frame=frame, preview=True) # must
        print(f"result: {predict}")
        # --------------- preview ---------------
        if not preview_images:
            continue
        # print(len(preview_images))
        cv2.imshow('origin', preview_images[0])
        cv2.imshow('openpose result', preview_images[1])
        cv2.imshow('dataset preview', preview_images[2])
        if cv2.waitKey(1) == 27:
                break
        # --------------- preview ---------------
    print(f"[{datetime.datetime.now()}] end: judgement")