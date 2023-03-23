from __future__ import annotations
import os
import cv2
import sys
import json
import copy
import time
import datetime
import pickle
import numpy as np

from command.data.src import util
from command.data.src.body import Body

class Dataset:
    def __init__(self, dataset_path:str, resize_shape:tuple, model_path:str='../model/openpose/body_pose_model.pth', category_list:list=['hands_up', 'hands_wave', 'others'], basic_fps:int=30, ajust_fps:int=10, **setting) -> None:
        self.setting = setting
        self.setting['dataset_path'] = dataset_path
        self.setting['model_path'] = model_path
        self.setting['resize_shape'] = resize_shape
        self.setting['category_list'] = category_list
        onehot = [ [1 if i==j else 0 for i in range(len(category_list))] for j in range(len(category_list))]
        self.setting['onehot'] = { category: onehot[i]  for i,category in enumerate(category_list)}
        self.setting['fps'] = basic_fps*int(dataset_path[-1])
        self.setting['ajust_fps'] = ajust_fps
        self.setting['ajust_counter'] = basic_fps/(ajust_fps)
        self._setting_path = os.path.join(self.setting['dataset_path'], 'setting.json')
    def make_dataset_for_openpose(self, review:bool=True, save:bool=True, save_type:str='pickle') -> dict:
        "dataset shape(output): (None, 30, 18, 2) "
        if save:
            self.setting['save_path'] = self.setting['dataset_path']
            self.setting['save_type'] = save_type
            self.dataset_type = f"data_f{self.setting['fps']}_af{self.setting['ajust_fps']}_rs{self.setting['resize_shape'][0]}-{self.setting['resize_shape'][1]}"
            self.setting['x_data_save_path'] = os.path.join(self.setting['save_path'], f"x_{self.dataset_type}")
            self.setting['y_data_save_path'] = os.path.join(self.setting['save_path'], f"y_{self.dataset_type}")
        self.data = {}
        x_data = []
        y_data = []
        print(self.setting)
        body_estimation = Body(self.setting['model_path'])
        for category_folder_name in os.listdir(self.setting['dataset_path']):
            category_folder_path = os.path.join(self.setting['dataset_path'], category_folder_name)
            if not os.path.isdir(category_folder_path):
                continue
            for mov_name in os.listdir(category_folder_path):
                mov_data = []
                mov_path = os.path.join(category_folder_path, mov_name)
                self.cap = cv2.VideoCapture(mov_path)
                if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) != self.setting['fps']:
                    continue
                start_time = time.time()
                counter = 1
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, dsize=self.setting['resize_shape'])
                    candidate, subset = body_estimation(frame)
                    parts_point_list = self._make_parts_point_list(candidate_list=candidate, subset_list=subset)
                    if counter == self.setting['ajust_counter']:
                        mov_data.append(np.array(parts_point_list))
                        counter = 0
                    if review:
                        image = util.draw_bodypose(frame, candidate, subset)
                        cv2.putText(
                            image, "FPS: %f" % (1.0 / (time.time() - start_time)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                        cv2.imshow('result', image)
                        if cv2.waitKey(1) == 27:
                            break
                    start_time = time.time()
                    counter += 1
                self.cap.release()
                x_data.append(np.array(mov_data))
                y_data.append(np.array(self.setting['onehot'][category_folder_name]))
                print(f"[{datetime.datetime.now()}] {category_folder_name}: {os.path.splitext(mov_name)[0].zfill(7)}{os.path.splitext(mov_name)[1]} [x_data_shape: {np.array(x_data).shape}] [y_data: {self.setting['onehot'][category_folder_name]}]")
        x_data = np.array(x_data, dtype=np.float64)
        y_data = np.array(y_data, dtype=np.float64)
        cv2.destroyAllWindows()
        if save:
            if self.setting['save_type'] == 'pickle':
                with open(f"{self.setting['x_data_save_path']}.pickle", 'wb') as fobj:
                    pickle.dump(x_data, fobj)
                with open(f"{self.setting['y_data_save_path']}.pickle", 'wb') as fobj:
                    pickle.dump(y_data, fobj)
            if self.setting['save_type'] == 'numpy':
                np.save(f"{self.setting['x_data_save_path']}.npy", x_data)
                np.save(f"{self.setting['y_data_save_path']}.npy", x_data)
            if 'setting.json' in os.listdir(self.setting['dataset_path']):
                with open(self._setting_path) as fobj:
                    setting = json.loads(fobj.read())
                if 'dataset_setting' in setting.keys():
                    setting['dataset_setting'][self.dataset_type] = self.setting
                    setting['dataset_setting']['datetime'][self.dataset_type] = str(datetime.datetime.now())
                else:
                    setting = {
                        'dataset_setting':
                            {self.dataset_type: self.setting}
                    }
                    setting['dataset_setting']['datetime'] = {self.dataset_type: str(datetime.datetime.now())}
            # else:
            #     setting = {
            #         'dataset_setting':
            #             {self.dataset_type: self.setting}
            #     }
            #     setting['dataset_setting']['datetime'] = [f'{self.dataset_type}_{str(datetime.datetime.now())}']
            with open(self._setting_path, 'w') as fobj:
                json.dump(setting, fobj, indent=4)
        self.data['x_data'] = x_data
        self.data['y_data'] = y_data
        return self.data
    def split_dataset(self, data:dict=None, split_rate:float=0.2) -> tuple:
        if not data:
            data = self.data
        train_dataset = {}
        train_dataset['x_data'], train_dataset['y_data'] = [], []
        val_dataset = {}
        val_dataset['x_data'], val_dataset['y_data'] = [], []
        counter = 0
        for i in range(len(data['x_data'])):
            if counter == int(1/split_rate-1):
                val_dataset['x_data'].append(copy.deepcopy(data['x_data'][i]))
                val_dataset['y_data'].append(copy.deepcopy(data['y_data'][i]))
                counter = 0
                continue
            else:
                train_dataset['x_data'].append(copy.deepcopy(data['x_data'][i]))
                train_dataset['y_data'].append(copy.deepcopy(data['y_data'][i]))
            counter += 1
        val_dataset['x_data'], val_dataset['y_data'] = np.array(val_dataset['x_data']), np.array(val_dataset['y_data'])
        train_dataset['x_data'], train_dataset['y_data'] = np.array(train_dataset['x_data']), np.array(train_dataset['y_data'])
        return train_dataset, val_dataset
    def load(self, x_path:str=None, y_path:str=None) -> tuple:
        if not x_path or y_path:
            with open(self._setting_path, 'r') as f:
                setting = json.loads(f.read())['dataset_setting']
                latest_time = None
                for data_type, date in setting['datetime'].items():
                    time = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
                    if not latest_time or latest_data < time:
                        latest_data = time
                        x_path = setting[data_type]['x_data_save_path']
                        y_path = setting[data_type]['y_data_save_path']
        try:
            with open(f'{x_path}.pickle', 'rb') as f:
                x_data = pickle.load(f)
            with open(f'{y_path}.pickle', 'rb') as f:
                y_data = pickle.load(f)
            # if self.setting['save_type'] == 'pickle':
            #     with open(f"{self.setting['x_data_save_path']}.pickle", 'rb') as f:
            #         x_data = pickle.load(f)
            #     with open(f"{self.setting['y_data_save_path']}.pickle", 'rb') as f:
            #         y_data = pickle.load(f)
            # elif self.setting['save_type'] == 'numpy':
            #     x_data = np.load(f"{self.setting['x_data_save_path']}.npy")
            #     y_data = np.load(f"{self.setting['y_data_save_path']}.npy")
            self.data = {
                'x_data': x_data,
                'y_data': y_data
            }
            return self.data
        except FileNotFoundError as e:
            print(f"[{datetime.datetime.now()}] {e}")
            sys.exit()
    def _make_parts_point_list(self, candidate_list:list, subset_list:list) -> list:
        parts_point_list = [ [-1, -1] for _ in range(18) ]
        if len(subset_list) == 0:
            return np.array(parts_point_list)
        subset = subset_list[0]
        for j, parts_num in enumerate(subset[:-2]):
            point_list = [-1,-1]
            if parts_num != -1:
                for candidate in candidate_list:
                    if parts_num == candidate[3]:
                        point_list = [candidate[0], candidate[1]]
                        break
            parts_point_list[j] = np.array(point_list)
        return np.array(parts_point_list)
    def _make_parts_point_list(self, candidate_list:list, subset_list:list, limitation:bool=True) -> list:
        parts_point_list = [ [-1, -1] for _ in range(18) ]
        if limitation:
            if len(subset_list) == 0:
                return parts_point_list
            subset = subset_list[0]
            for j, parts_num in enumerate(subset[:-2]):
                point_list = [-1,-1]
                if parts_num != -1:
                    for candidate in candidate_list:
                        if parts_num == candidate[3]:
                            point_list = [candidate[0], candidate[1]]
                            break
                parts_point_list[j] = np.array(point_list)
        else:
            for i, subset in enumerate(subset_list):
                list = []
                for j, parts_num in enumerate(subset[:-2]):
                    point_list = [None,None]
                    if parts_num != -1:
                        for candidate in candidate_list:
                            if parts_num == candidate[3]:
                                point_list = [candidate[0], candidate[1]]
                                break
                    list.append(np.array(point_list))
                parts_point_list[j] = np.array(list)
        return np.array(parts_point_list)
    def __dell__(self) -> None:
        if self.cap:
            self.cap.release()