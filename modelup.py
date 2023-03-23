from __future__ import annotations
import logging

import os
import json
import numpy as np
import time
import datetime

import command
from command.data.split_mov import split_all_mov
from command.data.make_dataset import Dataset
from command.data.preprocessing import *
from command.model.basic_model import TransformerModel

MODEL_SETTING_PATH = os.path.join('model', 'setting.json')

# ログ出力設定
logger = logging.getLogger('ModelUp')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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
    # path設定
    origin_data_path = r'setup\data\origin'
    openpose_model_path = r'model\openpose\body_pose_model.pth'
    category = [ c for c in os.listdir(origin_data_path) if os.path.isdir(os.path.join(origin_data_path, c))]
    # origin data の分割 (分岐作成予定)
    logger.info('start: split mov')
    split_data_dir_path = split_all_mov(data_path=origin_data_path, split_time=SPLIT_TIME)
    # split_data_dir_path = r'C:\Users\oden1\repAIn\FlowLineAnalyzer\gesturedetection\setup\data\dataset\time_3'
    logger.info('end: split mov')
    # 骨格推定時系列データの作成
    logger.info('start: make dataset')
    dataset = Dataset(
        dataset_path=split_data_dir_path,
        model_path=openpose_model_path,
        resize_shape=(WIDTH, HEIGHT),
        ajust_fps=AJUST_FPS,
        category_list=category
    )
    data = dataset.make_dataset_for_openpose() # making dataset
    # data = dataset.load() # loading dataset
    # データ正規化
    data['x_data'] = del_legs(data=data['x_data'])
    data['x_data'] = coordinate_transformation(data=data['x_data'], width=WIDTH, height=HEIGHT)
    data['x_data'] = normalization(data=data['x_data'], max_x=WIDTH, max_y=HEIGHT)
    train_data, val_data = dataset.split_dataset(data=data, split_rate=DATASET_SPLIT_RATE)
    input_shape = train_data['x_data'].shape[1:]
    logger.info('end: make dataset')
    # モデル定義
    logger.info('start: model up')
    model = TransformerModel()
    # for lr in np.arange(0.0006, 0.001, 0.00005):
    model.modelup(
        model_name=MODEL_NAME,
        input_shape=input_shape,
        head_size=256,
        num_heads=4,
        filter_dim=8,
        num_transfomer_blocks=4,
        trans_dropout=0.4,
        mlp_unit=[256, 128, 64],
        mlp_dropout=[0.4, 0.4, 0.25],
        learning_rate=0.00065,
        category_list=category
    )
    model.summary()
    # モデル学習
    history = model.train(
        dataset=train_data,
        val_dataset=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    logger.info('end: model up')