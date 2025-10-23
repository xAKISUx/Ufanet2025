import pandas as pd
import numpy as np
from torch import randperm
import os
from tqdm import tqdm

TRAIN_SPLIT = 0.75
COLS = ["path", "label"]
FOLDERS_PATH = "../datasets/Faces"

def preprocess(df:pd.DataFrame):
    """
    Переименовывает колонки path_to_folder в path и name в label. Также меняет начальный путь на путь относительно директории датасета.\n\n
    Возвращает измененный датафрейм.
    """
    df = df.rename(columns={"path_to_folder": "path", "name": "label"})
    df["path"] = df["path"].apply(lambda x: f"{FOLDERS_PATH}/{x}")
    return df

df_rf = preprocess(pd.read_csv("../datasets/Faces/ussr_russia.csv", sep=";"))
df_usa = preprocess(pd.read_csv("../datasets/Faces/usa.csv", sep=";"))
df_blog = preprocess(pd.read_csv("../datasets/Faces/bloggers.csv", sep=";"))

df = pd.DataFrame(None, columns=COLS) #создается общий дф
df_desc = pd.DataFrame(None, columns=["label", "link","specialization","films_years","description"]) #дф с описанием людей
df = pd.concat((df, df_rf[COLS], df_usa[COLS], df_blog[COLS])) #объединение изначальных дф
df_desc = pd.concat((df_desc, df_rf[["label", "link", "specialization", "films_years", "description"]],
                     df_usa[["label", "link", "specialization", "films_years", "description"]], 
                     df_blog["label"])) #объединение описаний изначальных дф

#здесь уже идет сохрание полного пути до картинки
#посльку в дф указан путь к папке с картинками определенного человека
arr = df.to_numpy()
result = []
for line in tqdm(arr):
    imgs = os.listdir(line[0])
    imgs = [i for i in imgs if ".jpg" in i]
    for img in imgs:
        result += [(f"{line[0]}/{img}", line[1])]

arr = np.array(result)
arr = arr[randperm(df.shape[0])] #перемешиваем массив
train = pd.DataFrame(arr[:int(TRAIN_SPLIT*df.shape[0])], columns=COLS) #создаем обучающую выборку
val = pd.DataFrame(arr[int(TRAIN_SPLIT*df.shape[0]):], columns=COLS) #создаем валидационную выборку

#сохраняем дф
train.to_csv("train.csv", index = False)
val.to_csv("val.csv", index = False)
df_desc.rename(columns={"label": "name"})
df_desc.to_csv("description.csv", index = False)