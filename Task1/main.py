from PIL import Image
import time
from rumodel import CLIP #ru
#from model import CLIP #eng
import os
import streamlit as st

IMGS_PATH = "../datasets/Text2Img/imgs"
NEW_MSG = lambda x, y, z: {"role": x, "content": y, "img": z} #функция для быстрого создания сообщения

#единожды выполняемая функция
@st.cache_resource
def init_model():
    model = CLIP()
    
    if not (os.path.exists("./metadata_ru.pkl") and os.path.exists("./indexes_ru.faiss")): #ru
    #if not (os.path.exists("./metadata.pkl") and os.path.exists("./indexes.faiss")): #eng
        imgs_srcs = os.listdir(IMGS_PATH)
        imgs_srcs = [f"{IMGS_PATH}/{i}" for i in imgs_srcs]
        model.build_indexes(imgs_srcs, 64)
        model.save_indexes()
    else:
        model.load_indexes()
        print(model.index.ntotal / len(os.listdir(IMGS_PATH)))
    
    return model

def get_response(promt):
        """Осуществляет красивый вывод сообщения"""
        response = f"Finded photo by ur query - {promt}"
        for word in response.split():
            yield f"{word} "
            time.sleep(0.05)

def main():
    model = init_model()
    
    #создание истории сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #выводим старые сообщения
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["content"] != None:
                st.markdown(msg["content"])
            if msg["img"] != None:
                st.image(msg["img"])
                
    if promt := st.chat_input("Enter"):
        with st.chat_message("user"):
            st.markdown(promt) 
            st.session_state.messages.append(NEW_MSG("user", promt, None)) #добавление сообщения в историю
        
        results = model.search_by_text(promt)
        with st.chat_message("assistant"):
            response = st.write_stream(get_response(promt))
            st.session_state.messages.append(NEW_MSG("assistant", response, None))  #добавление сообщения в историю
            for i in results:
                img = Image.open(i["src"])
                st.image(img)
                st.session_state.messages.append(NEW_MSG("assistant", None, img))  #добавление сообщения в историю
            
if __name__ == "__main__":
    main()