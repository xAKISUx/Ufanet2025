from transformer import ImgTransformer
from model import FaceClassifier
from datasets import CelebaDataset
import sys

def get_help():
    print("""Производит обучение модели. Имеет такие параметры как:
          --epochs\t\t| кол-во эпох (def: 100) 
          --batch_size\t\t| размер батча (def: 32)
          --val\t\t\t| валидация во время обучения True/False (def: True)
          --imgs_path\t\t| путь к папке с картинками (def: ../datasets/FacesWithAttrs/img_align_celeba/img_align_celeba)
          --dfs_path\t\t| путь к папке с данными csv (def: ../datasets/FacesWithAttrs)
          --save_path\t\t| путь для сохранения модели (def: ./resnet50.pt)
          --checkpoint_path\t| путь к чекпоинту(сохраненной модели) (def: None)""")

def is_bool(txt:str):
    txt = txt.lower()
    return True if txt == "true" else False

def parse_args(args):
    result = {
        "epochs": 100,
        "batch_size": 32,
        "imgs_path": "../datasets/FacesWithAttrs/img_align_celeba/img_align_celeba",
        "dfs_path": "../datasets/FacesWithAttrs",
        "val": "True",
        "save_path": "./resnet50.pt",
        "checkpoint_path": None
    }

    for i, arg in enumerate(args):
        if i % 2 == 0:
            key = arg.replace("--", "")
            condition = key in result.keys()
            if not condition:
                get_help()
            assert condition, f"there is no such argument: {arg}"
        else:
            condition = "-" not in arg
            if not condition:
                get_help()
            assert condition, "after arg name must be value"
            result[key] = arg
    return result

if __name__ == "__main__":
    args_raw = sys.argv[1:]
    args = parse_args(args_raw)
    transformer = ImgTransformer()
    model = FaceClassifier(transformer)
    dataset = CelebaDataset(args["dfs_path"], args["imgs_path"], transformer=transformer)
    print(args)
    if is_bool(args["val"]):
        val_ds = CelebaDataset(args["dfs_path"], args["imgs_path"], val=True, transformer=transformer)
    else:
        val_ds = None

    model.train(
        dataset, 
        val_ds=val_ds, 
        epochs=int(args["epochs"]), 
        batch_size=int(args["batch_size"]),
        checkpoint=args["checkpoint_path"]
        )
    model.save(args["save_path"])