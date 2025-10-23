import ruclip
from PIL import Image
import torch
import faiss
from tqdm import tqdm
import pickle 
import numpy as np

class CLIP:
    def __init__(self, model_name:str="ruclip-vit-base-patch32-384", dimension:int=512) -> None:
        self.imgs_srcs= None
        #модель была скачана https://huggingface.co/ai-forever/ruclip-vit-base-patch32-384/tree/main
        #все файлы модели были сохранены в папке rumodels
        self.model = ruclip.CLIP.from_pretrained("./rumodels")
        self.preprocess = ruclip.RuCLIPProcessor.from_pretrained("./rumodels")
        self.model.to("cuda")
        self.model.eval()
        self.index = faiss.IndexFlatIP(dimension)
    
    def build_indexes(self, srcs:list, batch_size=32):
        self.imgs_srcs = np.array(srcs)
        for i in tqdm(range(0, len(self.imgs_srcs), batch_size)):
            batch_srcs = srcs[i:i+batch_size]
            batch_imgs = []
            for src in batch_srcs:
                try:
                    img = self.preprocess(images=[Image.open(src)])["pixel_values"].to("cuda")
                    batch_imgs += [img]
                except Exception as e:
                    print(f"Не вышло загрузить: {src} - {e}")

            img_features = self.__get_img_features(batch_imgs)
            print(img_features.shape)
            self.index.add(img_features)
    
    def __get_text_features(self, text:str):
        """Возвращает нормализованные фичи текста"""
        prepared = self.preprocess([text])
        ids = prepared["input_ids"].to("cuda")
        with torch.no_grad():
            text_features = self.model.encode_text(ids)
            text_features /= torch.norm(text_features, dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype("float32") 
    
    def __get_img_features(self, imgs):
        """Возвращает нормализованные фичи картинок"""
        with torch.no_grad():
            tensor = torch.cat(imgs)
            img_features = self.model.encode_image(tensor)
            img_features /= torch.norm(img_features, dim=-1, keepdim=True)
            
        return img_features.cpu().numpy().astype("float32")    
    
    def search_by_text(self, text:str, k=5):
        """Производит поиск и возвращает результаты поиска"""
        query = self.__get_text_features(text)
        distances, indices = self.index.search(query, k)
        results = []
        for i in range(len(indices[0])):
            results += [{
                "src": self.imgs_srcs[indices[0][i]],
                "distance": distances[0][i]
            }]
            
        return results
    
    def save_indexes(self, path="./"):
        """Производит сохранение бд и метаданных"""
        metadata = {
            "imgs_srcs": self.imgs_srcs
        }
        faiss.write_index(self.index, f"{path}indexes_ru.faiss")
        with open(f"{path}metadata_ru.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_indexes(self, path="./"):
        """Производит загрузку бд и метаданных"""
        self.index = faiss.read_index(f"{path}indexes_ru.faiss")
        with open(f"{path}metadata_ru.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.imgs_srcs = metadata["imgs_srcs"]
        