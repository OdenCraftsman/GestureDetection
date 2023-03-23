from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import time
import datetime

import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import optimizers
from keras.optimizers import Adam

class TransformerModel:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._model_dir = str(self._base_path / Path("model/gesture"))
        self._setting_path = os.path.join(self._model_dir, 'setting.json')
        self._result_dir = str(self._base_path / Path("setup/result"))
        self.setting = {}
    def modelup(
        self,
        model_name:str,
        input_shape:list,
        head_size:int=256,
        num_heads:int=4,
        filter_dim:int=8,
        num_transfomer_blocks:int=4,
        trans_dropout:float=0.4,
        mlp_unit:list=[256, 128, 64],
        mlp_dropout:list[float]=[0.4, 0.4, 0.25],
        loss:str='categorical_crossentropy',
        optimizer:str='adam',
        learning_rate:float=0.0008,
        accuracy:list[str]=['accuracy'],
        category_list:list=['hands_up', 'hands_wave', 'others'],
        **setting
    ) -> None:
        # 初期設定
        counter = self._model_counter(dir=self._model_dir)
        self.model_base_name = model_name
        self.model_name = f"{model_name}_no{counter}"
        # self.setting = setting
        self.setting = {}
        self.setting['model_path'] = os.path.join(self._model_dir, self.model_name)
        self.setting['model_detaile'] = {
            'model_base_type': 'Transformer',
            'input_shape': input_shape,
            'head_size': head_size,
            'num_heads': num_heads,
            'filter_dim': filter_dim,
            'num_transfomer_blocks': num_transfomer_blocks,
            'trans_dropout': trans_dropout,
            'mlp_unit': mlp_unit,
            'mlp_dropout': mlp_dropout,
            'loss': loss,
            'optimizer': optimizer,
            'learning_rate': learning_rate
        }
        self.setting['accuracy'] = accuracy
        self.setting['category_list'] = category_list
        self.setting['num_category'] = len(category_list)
        x = input = keras.Input(shape=input_shape)
        for _ in range(num_transfomer_blocks):
            # attention and normalization
            x = layers.MultiHeadAttention(
                key_dim=head_size,
                num_heads=num_heads,
                dropout=trans_dropout
            )(x,x)
            x = layers.Dropout(trans_dropout)(x)
            x = layers.LayerNormalization(epsilon=1e-4)(x)
            res = x + input
            # feedfoward
            x = layers.Conv2D(filters=filter_dim, kernel_size=1, activation='relu')(res)
            x = layers.Dropout(trans_dropout)(x)
            x = layers.Conv2D(filters=input.shape[-1], kernel_size=1)(x)
            x = layers.Dropout(trans_dropout)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = res + x
        x = layers.Flatten()(x)
        for dim, dropout in zip(mlp_unit, mlp_dropout):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-4)(x)
        output = layers.Dense(len(category_list), activation='softmax')(x)
        self.model = keras.Model(input, output)
        self.model.compile(
            loss=loss,
            optimizer=optimizers.Adam(learning_rate=learning_rate), #RMSprop(learning_rate=0.001), #,
            metrics=accuracy
        )
        return self.model
    def load_model(self, model_name:str=None) -> None:
        if not model_name:
            if not self.setting:
                with open(self._setting_path, 'r') as fobj:
                    print(self._setting_path)
                    setting = json.loads(fobj.read())
                latest_time = None
                for model_type, date in setting['datetime'].items():
                    time = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
                    if not latest_time or latest_data < time:
                        latest_data = time
                        latest_model_type = model_type
                self.setting = setting[latest_model_type]
        else:
            with open(self._setting_path, 'r') as fobj:
                print(self._setting_path)
                setting = json.loads(fobj.read())
            self.setting = setting[model_name]
        model_path = Path(f'{self.setting["model_path"]}.h5')
        print(model_path)
        self.model = keras.models.load_model(str(self._base_path / model_path))
    def train(self, dataset:dict, val_dataset:dict, epochs:int=100, batch_size:int=64) -> keras.callbacks.History:
        counter = self._model_counter(dir=self._model_dir)
        self.model_name = f"{self.model_base_name}_no{counter}"
        self.setting['model_path'] = os.path.join(self._model_dir, self.model_name)
        self._preview_path = f"{self.model_name}_epochs{epochs}_batch{batch_size}"
        print(self.setting)
        self.history = self.model.fit(
                x=dataset['x_data'],
                y=dataset['y_data'],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_dataset['x_data'], val_dataset['y_data']),
                shuffle=True
            )
        self._save(self.setting["model_path"])
        setting = {}
        if self._setting_path in os.listdir(self._model_dir):
            with open(self._setting_path) as fobj:
                setting = json.load(fobj.read())
        else:
            setting['datetime'] = {}
        setting[self.model_name] = self.setting
        setting['datetime'][self.model_name] = str(datetime.datetime.now())
        with open(self._setting_path, 'w') as fobj:
            json.dump(setting, fobj, indent=4)
        self._preview_history()
        return self.history
    def detection(self, data:np.array) -> str:
        result = self.model.predict(data)[0]
        index = result.argmax()
        predict = self.setting['category_list'][index]
        print(f'[{datetime.datetime.now()}] [end: predict] [{predict}]')
        return predict, result
    def summary(self) -> None:
        self.model.summary()
    def _save(self, model_name:str) -> None:
        self.model.save(f'{model_name}.h5')
    def _preview_history(self) -> None:
        metric = self.setting['accuracy']
        os.makedirs(self._result_dir, exist_ok=True)
        for i, m in enumerate(metric):
            self.result_save_path = os.path.join(self._result_dir, f"{self.model_name}_{i}")
            fig = plt.figure(figsize=(12,4))
            fig.suptitle(f"lr{self.setting['model_detaile']['learning_rate']}")
            ax1 = fig.add_subplot(1,2,1)
            ax1.plot(self.history.history[m], label=m)
            ax1.plot(self.history.history[f"val_{m}"], label=f"val_{m}")
            ax1.set_xlabel('epochs[]')
            ax1.set_ylabel(f'{m}[%]')
            ax1.legend()
            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(self.history.history['loss'], label='loss')
            ax2.plot(self.history.history['val_loss'], label='val_loss')
            ax2.set_xlabel('epochs[]')
            ax2.set_ylabel('loss[%]')
            ax2.legend()
            fig.tight_layout()
            fig.savefig(f"{self.result_save_path}.jpg")
    def _model_counter(self, dir:str='../../model/gesture') -> int:
        model_counter = 0
        for file in os.listdir(dir):
            if not os.path.splitext(file)[1] == '.h5':
                continue
            model_counter += 1
        return model_counter