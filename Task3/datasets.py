from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple, List
import os 

class CelebaDataset(Dataset):
    def __init__(self, dfs_path:str="./", imgs_path:str="./", batch_size:int=32, val:bool=False, transformer=None):
        self.transform = transformer
        self.val = val
        self.imgs_path = imgs_path
        self.batch_size = batch_size
        self.boxes, self.y, self.parts = self.__init_dfs(dfs_path, val)
        self.attrs = self.y.drop(columns=["image_id"]).columns
        if val:
            self.boxes = self.__val(self.boxes)
            self.y = self.__val(self.y)
            
            
    
    def __init_dfs(self, dfs_path:str, val:bool) -> Tuple[pd.DataFrame|None, pd.DataFrame, pd.DataFrame]:
        """Подготавливает df к работе с ними"""
        assert type(dfs_path) == str, "the dfs_path value must be string"

        parts = pd.read_csv(f"{dfs_path}/list_eval_partition.csv")#[160000:]
        y = pd.read_csv(f"{dfs_path}/list_attr_celeba.csv")#[160000:]
        if not os.path.exists(f"{dfs_path}/bbox.csv"):
            pd.read_csv("./bbox.csv").to_csv(f"{dfs_path}/bbox.csv") #костыль

        parts = parts.drop("image_id", axis=1)

        boxes = pd.read_csv(f"{dfs_path}/bbox.csv")#[160000:]
        boxes = boxes.rename(columns={"src": "image_id"})
        boxes, y = self.__dropna(boxes, y)
        
        return boxes, y, parts      
    
    def __dropna(self, x:pd.DataFrame, y:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Удаляет строки, на которых нету координатов box.\n
        Возвращает x(датафрейм с координатами box), y(атрибуты лица) 
        """
        temp = pd.concat((x, y.drop(columns=["image_id"])), axis=1)
        temp = temp.dropna(axis=0, ignore_index=True)
        temp = temp.reset_index(drop=True)
        return temp[x.columns], temp.drop(columns=x.columns[1:]) # для y выбрасываем все колонки x кроме image_id

    def get_sample_weights(self) -> np.ndarray:
        """ 
        Определяет насколько важный сампл. Создана для решения проблемы дисбаланса атрибутов. Насколько полезна и верна не проверялось\n
        Возвращает массив весов для каждого сампла
        """
        sample_weights = []
        frequency = {}
        y = pd.concat((self.y, self.parts), axis=1)
        y = y[y.partition == 0]
        y = y.drop(columns=["partition", "image_id"])
        for col in self.attrs:
            t_count = y[col][y[col] == 1].sum()
            frequency[col] = {
                "T": t_count / y.shape[0],
                "F": (y.shape[0] - t_count) / y.shape[0]
            }

        for id, row in y.iterrows():
            weight = 0
            attrs = 0
            for attr, value in row.items():
                weight +=  1 / frequency[attr]["T" if value == 1 else "F"]
                attrs += 1
            sample_weights.append(weight / attrs)

        sample_weights = np.array(sample_weights)
        sample_weights /= sample_weights.sum()
        return sample_weights

    def __val(self, df) -> pd.DataFrame:
        """
        Позволяет получить данные для валидации
        """
        buff = pd.concat((df, self.parts), axis=1)
        buff = buff[buff.partition == 1]
        return buff.drop(columns=["partition"]).reset_index(drop=True)

    def __get_crop_points(self, index) -> List[int]:
        """Возвращает координаты начальной и конечной точек главной диагонали для кропа"""
        box = self.boxes.iloc[index, 1:]
        return box.tolist()

    def __len__(self):
        if not self.val:
            return self.parts[self.parts.partition == 0].shape[0]
        return self.parts[self.parts.partition == 1].shape[0]
    
    def __getitem__(self, index):
        img = Image.open(f"{self.imgs_path}/{self.y.image_id[index]}")
        img = img.crop(self.__get_crop_points(index)) 
        x = self.transform(img)
        y = (self.y.iloc[index, 1:].to_numpy() + 1) / 2
        return x, y.astype(np.float32)