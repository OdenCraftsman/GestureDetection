from __future__ import annotations
import os
import sys
import json
import logging
import datetime

import cv2
import numpy as np

def split_all_mov(data_path:str, split_time:int=3) -> str:
    # ログ設定
    logger = logging.getLogger('SplitAllMov')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # ディレクトリパスおよび詳細設定
    dir_path = os.path.abspath(os.path.join(data_path, '..'))
    dataset_dir_path = os.path.join(os.path.join(dir_path, 'dataset'), f'time_{split_time}')
    setting_path = os.path.join(dataset_dir_path, 'setting.json')
    category_num = len(os.listdir(data_path))
    category = os.listdir(data_path)
    if not os.path.basename(dataset_dir_path) in os.listdir(dir_path):
        os.makedirs(dataset_dir_path, exist_ok=True)
        setting = {
            'data_path': data_path,
            'split_time': 3,
            'category': category,
            'category_num': category_num,
            'cap_detaile': {},
            'splitted_mov_num': {},
            'pickle_data':[]
        }
    else:
        with open(setting_path) as fobj:
            setting = json.load(fobj.read())
    # 動画分割
    for category_name in os.listdir(data_path):
        category_dir_path = os.path.join(data_path, category_name)
        if not os.path.isdir(category_dir_path):
            continue
        save_category_dir_path = os.path.join(dataset_dir_path, category_name)
        if not category_name in os.listdir(dataset_dir_path):
            os.makedirs(save_category_dir_path)
        counter = 0
        for mov_name in os.listdir(category_dir_path):
            if not os.path.splitext(mov_name)[1] == '.mp4':
                continue
            # print(f"[{datetime.datetime.now()}] [split_mov] ")
            mov_path = os.path.join(category_dir_path, mov_name)
            cap = cv2.VideoCapture(mov_path)
            fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_time = video_frame_num/video_fps
            video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            setting['cap_detaile'][mov_path] = {
                'fps': video_fps,
                'frame': video_frame_num,
                'time': video_time,
                'width': video_width,
                'height': video_height
            }
            for i in range(int(video_time)):
                logger.info(f'[{category_name}] [{mov_name}] {counter:6}.mp4')
                save_path = os.path.join(save_category_dir_path, f'{counter}.mp4')
                output_cap = cv2.VideoWriter(
                    save_path,
                    fourcc,
                    int(video_fps),
                    (int(video_width), int(video_height))
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, i*video_fps)
                for _ in range(i*int(video_fps), (i+split_time)*int(video_fps)):
                    ret, frame = cap.read()
                    output_cap.write(frame)
                output_cap.release()
                counter += 1
            cap.release()
    if 'setting.json' in os.listdir(dataset_dir_path):
        with open(setting_path, 'r') as fobj:
            save_setting = json.load(fobj.read())
        save_setting['split_setting'] = setting
    else:
        save_setting = {
            'split_setting': setting
        }
    with open(setting_path, 'w') as fobj:
        json.dump(save_setting, fobj, indent=2)
    return dataset_dir_path

def split_mov(mov_path:str, category:str, split_time:int=3):
    pass

    # origin_path = os.path.join(data_path, origin_name)
    # if not os.path.splitext(origin_name)[1] == 'mp4':
    #     continue
    # save_category_dir_path = os.path.join(dataset_dir_path, os.path.splitext(origin_name)[0])
    # if not os.path.splitext(origin_name)[0] in os.listdir(dataset_dir_path):
    #     os.makedirs(save_category_dir_path)
    # cap = cv2.VideoCapture(origin_path)
    # fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # video_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # video_time = video_frame_num/video_fps
    # video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # setting['cap_detaile'][origin_name] = {
    #     'fps': video_fps,
    #     'frame': video_frame_num,
    #     'time': video_time,
    #     'width': video_width,
    #     'height': video_height
    # }
    # for i in range(int(video_time)):
    #     logger.info(f'{os.path.splitext(origin_name)[0]}: {i}.mp4')
    #     save_path = os.path.join(save_category_dir_path, f'{i}.mp4')
    #     output_cap = cv2.VideoWriter(
    #         save_path,
    #         fourcc,
    #         int(video_fps),
    #         (int(video_width), int(video_height))
    #     )
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i*video_fps)
    #     for _ in range(i*int(video_fps), (i+split_time)*int(video_fps)):
    #         ret, frame = cap.read()
    #         output_cap.write(frame)
    #     output_cap.release()
    # cap.release()