from insightface.app import FaceAnalysis
import cv2
import sys
from pymilvus import MilvusClient
import streamlit as st
import time
from PIL import Image
import numpy as np
import pandas as pd

#какая-то библиотека ругалась на отсутствие warnings, после данной конструкции предупреждение пропало
if 'warnings' not in sys.modules:
    import warnings
    sys.modules['warnings'] = warnings

#выполнение следующей функции единожды
@st.cache_resource
def init():
    model = FaceAnalysis("buffalo_l", verbose=0)
    model.prepare(0, det_size=(512,512))
    client = MilvusClient("face_security.db")

    return model, client

def load_df():
    """Открывает дф с описанием людей"""
    #почему не descripotion.csv? либо я о нем забыл, либо была какая то причина
    df_russia = pd.read_csv("../datasets/Faces/ussr_russia.csv", sep=";")
    df_usa = pd.read_csv("../datasets/Faces/usa.csv", sep=";")
    df_bloggers = pd.read_csv("../datasets/Faces/bloggers.csv", sep=";")
    df = {
        "russia": df_russia,
        "usa": df_usa,
        "bloggers": df_bloggers
    }

    return df

def search(img, model:FaceAnalysis, client:MilvusClient, df):
    """Осуществляет поиск по бд. Возвращает наиболее подходящего человека (По мнению модели)"""
    embed = model.get(img, 1)[0]["embedding"]
    #осуществление поиска по кос расстоянию
    search_result = client.search(
        collection_name="security",
        data=[embed],
        output_fields=["name"],
        search_params={"metric_type": "COSINE"},
        limit=1
    )[0][0]["name"]
    desc = df["bloggers"][df["bloggers"]["name"] == search_result]

    #далее формирование ответа
    if desc.shape[0] != 0:
        result = f"Это блоггер {search_result}"
    else:
        desc_russia = df["russia"][df["russia"]["name"] == search_result]
        desc_usa = df["usa"][df["usa"]["name"] == search_result]
        desc = desc_russia if desc_russia.shape[0] != 0 else desc_usa
        desc = desc.to_numpy()[0]
        result = f"##### Это актер [{search_result}]({desc[2]}).  \n"
        if not pd.isna(desc[3]):
            result += "**Занимал(а) такие роли как:**  \n" + "".join([f"- {i}  \n" for i in desc[3].split(",")]) + "  \n"
        else:
            result += f"Занимал(а) такие роли как:** \n- нет данных"
        if not pd.isna(desc[5]):
            result += f"**Снимал(ся/ась) в следующих фильмах:**  \n" + "".join([f"- {i}  \n" for i in desc[5].split(",")])
        else:
            result += f"**Снимал(ся/ась) в следующих фильмах:** \n- нет данных"
    
    return result

def main():
    model, client = init()
    DF = load_df() #для формирования ответа, содержит описания каждого человека
    with st.chat_message("user"):
        user_files = st.file_uploader(
            "Updload image", 
            type=["jpg", "jpeg", "png"], 
            help="supported jpg, jpeg, png"
        )
        if user_files != None:
            img = np.array(Image.open(user_files))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)
    
    with st.chat_message("assistant"):
        if user_files != None:
            result = search(img, model, client, DF)
            st.markdown(result)

if __name__ == "__main__":
    main()