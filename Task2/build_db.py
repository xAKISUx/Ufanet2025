from insightface.app import FaceAnalysis
import cv2
from pymilvus import MilvusClient
import os 
import pandas as pd
from tqdm import tqdm
DIRS_PATH = "../datasets/Faces"

model = FaceAnalysis("buffalo_l", verbose=0)
model.prepare(0,det_size=(512,512))
client = MilvusClient("face_security.db")
if client.has_collection("security"):
    client.drop_collection("security")
client.create_collection(
    collection_name="security",
    vector_field_name="vector",
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE"
)

#проходимся по директориям (блогеры, рф_ссср, сша)
dirs = os.listdir(DIRS_PATH)
dirs = [i for i in dirs if os.path.isdir(f"{DIRS_PATH}/{i}")]
for dir in dirs:
    path = f"{DIRS_PATH}/{dir}"
    df = pd.read_csv(f"{path}.csv", sep=";")
    df["path_to_folder"] = df["path_to_folder"].apply(lambda x: x[7:] if "actors" in x else x) #папки я вручную переименовал (почему? я не помню), вообщем было убрано слово actors_
    actor_dirs = os.listdir(path)
    actor_dirs = [i for i in actor_dirs if os.path.isdir(f"{path}/{i}")]
    for actor_dir in tqdm(actor_dirs, desc=f"from {dir}"): #проходимся по всем директориям актеров
        filepath = f"{DIRS_PATH}/{dir}/{actor_dir}"
        files = os.listdir(filepath)
        files = [f"{filepath}/{i}" for i in files if ".jpg" in i]
        for f in files: #проходимся по всем фотографиям актера
            df_filepath = f"{dir}/{actor_dir}"
            img = cv2.imread(files[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            find_result =  model.get(img, 1) 
            if len(find_result) != 0: #берем только первую фотографию
                break
        if len(find_result) == 0: #не удалось найти лицо ни наодной фотке - переход к следующему актеру
            continue
        embedding = find_result[0]["embedding"] #получаем эмбеддинг
        name = df["name"][df["path_to_folder"] == df_filepath] #получаем имя актера
        try:
            #записываем актера в бд
            name = name.to_numpy()[0]
            client.insert(
                collection_name="security",
                data={
                    "vector": embedding,
                    "name": name
                }
            )
        except Exception as e:
            print(e)