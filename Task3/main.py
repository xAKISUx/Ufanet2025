from model import FaceClassifier
from PIL import Image
import os
from transformer import ImgTransformer
import numpy as np
import streamlit as st
import cv2

@st.cache_resource
def init():
    transformer = ImgTransformer()
    model = FaceClassifier(transformer)
    model.load("./models/resnet50_e25.pt")
    if not os.path.exists("./imgs"):
        os.mkdir("./imgs")
    return model

def search(img, model: FaceClassifier):
    result = model.predict([img]) #получаем атрибуты лиц
    text = "Attrs: \n\n"
    for i, face in enumerate(result["./imgs/img.jpg"]):
        if i != 0:
            text += "\n\n"

        text += "".join(f"- {i}\n" for i in face)
    return text

def main():
    model = init()
    with st.chat_message("user"):
        user_files = st.file_uploader(
            "Updload image", 
            type=["jpg", "jpeg", "png"], 
            help="supported jpg, jpeg, png"
        )
        if user_files != None:
            img = Image.open(user_files)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)
            cv2.imwrite("./imgs/img.jpg", img) #костыль)

    with st.chat_message("assistant"):
        if user_files != None:
            result = search("./imgs/img.jpg", model)
            st.markdown(result)

if __name__ == "__main__":
    main()