from datasets import CelebaDataset
from ultralytics import YOLO
from PIL import Image
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet50
from tqdm import tqdm

class FaceClassifier:
    def __init__(self, transformer, num_classes:int=40):
        self.num_classes = num_classes
        self.face_finder = YOLO("./models/yolov12l-face.pt")
        self.classificator = resnet50(num_classes=num_classes).to("cuda")
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.classificator.parameters(), lr=1e-3, weight_decay=1e-4)
        self.transformer = transformer
        self.classes = None

    def train(self, ds:CelebaDataset, val_ds:CelebaDataset=None, epochs:int=50, batch_size:int=32, save=True, checkpoint:str=None):
        checkpoint_epoch = 0
        self.classes = ds.attrs
        samples_w = ds.get_sample_weights()
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=WeightedRandomSampler(samples_w, samples_w.shape[0])
        )

        if self.classificator.training == False:
            self.classificator.train()
        if checkpoint != None:
            checkpoint_epoch = self.__load_checkpoint(checkpoint)

        for e in range(checkpoint_epoch, epochs):
            predicted_cls = 0
            total_loss = 0
            for batch_x, batch_y in tqdm(dataloader, desc=f"epoch {e+1}/{epochs}"):
                batch_x = batch_x.to("cuda", non_blocking=True)
                batch_y = batch_y.to("cuda", non_blocking=True)
                
                y_pred = self.classificator(batch_x)
                predicted_cls += self.__success_pred(torch.sigmoid(y_pred), batch_y)
                loss = self.loss_fn(y_pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"loss: {total_loss/len(dataloader)}\t\t accuracy: {self.__accuracy(predicted_cls, ds):0.1f}%")
            if val_ds is not None:
                self.val(val_ds)
            if save:
                self.save(f"./resnet50_e{e+1}.pt")

    def __load_checkpoint(self, checkpoint:str):
        self.load(checkpoint)
        epoch = re.search(r"_e\d+.pt", checkpoint).group()
        if epoch is not None:
            epoch = re.search(r"\d+", epoch).group()

        return epoch if epoch is not None else 0

    def __accuracy(self, success, dataset):
        """
        Возвращает точность. Рассчитывается кол-во угаданных классов / (количество самплов * количество классов) * 100.\n\n
        Вообще точность надо как то иначе лучше рассчитывать
        """
        return success/(len(dataset)*self.num_classes)*100
    
    def __success_pred(self, y_pred, y):
        """Возвращает количество классов, которые были верно определены"""
        return sum([sum(y_pred[i].round() == y[i]) for i in range(y_pred.shape[0])])
    
    def val(self, ds:CelebaDataset):
        """Выводит результаты модели на валидационной выборке"""
        predicted_cls = 0
        total_loss = 0
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to("cuda", non_blocking=True)
                batch_y = batch_y.to("cuda", non_blocking=True)
                
                y_pred = self.classificator(batch_x)
                predicted_cls += self.__success_pred(torch.sigmoid(y_pred), batch_y)
                loss = self.loss_fn(y_pred, batch_y)
                total_loss += loss.item()
            print(f"val loss: {total_loss/len(dataloader)}\t\t val accuracy: {self.__accuracy(predicted_cls, ds):0.1f}%")

    def predict(self, imgs:str, threshold:float=0.5):
        """Возврщает результаты предсказаний модели"""
        _imgs = []
        predicts = {}
        faces = {
            "index": [],
            "crop": []
        }
        if self.classificator.training:
            self.classificator.eval()
        
        for img in imgs:
            _imgs.append(Image.open(img))
        
        with torch.no_grad():
            results = self.face_finder(imgs, verbose=False) #получаем результаты по каждой картинке
            for ir, result in enumerate(results): #проход по картинкам
                for box in result.boxes: #проход по лицам на картинке
                    faces["index"].append(ir) #сохранение индекса изображения
                    faces["crop"].append(self.transformer(_imgs[ir].crop(box.xyxy[0].tolist())).unsqueeze(0))

            if len(faces["crop"]) == 0:
                return None
                    
            crops = torch.cat(faces["crop"], dim=0).to("cuda")
            y = self.classificator(crops)
            y = torch.sigmoid(y) #получение вероятностей каждого класса

            for i in range(y.shape[0]):
                if imgs[faces["index"][i]] not in predicts.keys():
                    predicts[imgs[faces["index"][i]]] = [] #создание ключа с путем к изображению
                predicts[imgs[faces["index"][i]]].append(self.__get_class_name(y[i], threshold)) #запись в этот ключ атрибуты найденных лиц

        return predicts
    
    def __get_class_name(self, y, threshold=0.5):
        """Возвращает список названий классов, которые проходят границу (threshold)"""
        result = []
        for i in range(y.shape[0]):
            if y[i] < threshold:
                continue
            result.append(self.classes[i])

        return result

    def save(self, path="./resnet50.pt"):
        """Сохраняет модель, количество атрибутов и классы"""
        torch.save({
            "state_dict": self.classificator.state_dict(),
            "num_attributes": len(self.classes),
            "classes": self.classes
        }, path)

    def load(self, path="./resnet50.pt"):
        """Загружает модель, количество атрибутов и классы"""
        data = torch.load(path, weights_only=False)
        self.classes = data["classes"]
        self.num_classes = data["num_attributes"]
        self.classificator.load_state_dict(data["state_dict"])
        self.classificator.eval()