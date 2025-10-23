import open_clip
from PIL import Image
import torch
import faiss
from tqdm import tqdm
import pickle 
import numpy as np
import torch.nn.functional as F

class CLIP:
    def __init__(self, model_name:str="ViT-B-32", pretrained:str=None, dimension:int=512) -> None:
        self.imgs_srcs= None
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
        self.model.eval()
        self.index = faiss.IndexFlatIP(dimension)
    
    def build_indexes(self, srcs:list, batch_size=32):
        """Собирает бд"""
        self.imgs_srcs = np.array(srcs)
        imgs = []
        for i in tqdm(range(0, len(self.imgs_srcs), batch_size)):
            batch_srcs = srcs[i:i+batch_size]
            batch_imgs = []
            for src in batch_srcs:
                try:
                    img = self.preprocess(Image.open(src)).unsqueeze(0)
                    batch_imgs += [img]
                except Exception as e:
                    print(f"Не вышло загрузить: {src} - {e}")

                imgs += batch_imgs

        img_features = self.__get_img_features(imgs)
        self.index.add(img_features)
    
    def __get_text_features(self, text:str):
        """Возвращает нормализованные фичи текста"""
        prepared = self.tokenizer([text]) 
        with torch.no_grad(), torch.autocast("cuda"):
            text_features = self.model.encode_text(prepared)
            text_features /= torch.norm(text_features, dim=-1, keepdim=True)
        
        return text_features.numpy().astype("float32") 
    
    def __get_img_features(self, imgs):
        """Возвращает нормализованные фичи картинки"""
        with torch.no_grad(), torch.autocast("cuda"):
            tensor = torch.cat(imgs)
            img_features = self.model.encode_image(tensor)
            img_features /= torch.norm(img_features, dim=-1, keepdim=True)
            
        return img_features.numpy().astype("float32")    
    
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
        faiss.write_index(self.index, f"{path}indexes.faiss")
        with open(f"{path}metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_indexes(self, path="./"):
        """Производит загрузку бд и метаданных"""
        self.index = faiss.read_index(f"{path}indexes.faiss")
        with open(f"{path}metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.imgs_srcs = metadata["imgs_srcs"]
        